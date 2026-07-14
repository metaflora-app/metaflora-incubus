package installer

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var serviceModelID = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,127}$`)

type ServiceSpecOptions struct {
	InstallRoot string
	Executable  string
	ModelPath   string
	Host        string
	Port        int
	ModelID     string
}

type ServiceSpec struct {
	installRoot string
	executable  string
	modelPath   string
	host        string
	port        int
	modelID     string
	arguments   []string
	stdoutPath  string
	stderrPath  string
}

func NewServiceSpec(options ServiceSpecOptions) (ServiceSpec, error) {
	root := filepath.Clean(options.InstallRoot)
	executable := filepath.Clean(options.Executable)
	modelPath := filepath.Clean(options.ModelPath)
	if !filepath.IsAbs(root) || !withinRoot(executable, root) || !withinRoot(modelPath, root) {
		return ServiceSpec{}, errors.New("runtime paths must be absolute and managed")
	}
	if options.Host != "127.0.0.1" {
		return ServiceSpec{}, errors.New("runtime must bind to IPv4 loopback")
	}
	if options.Port < 1 || options.Port > 65535 {
		return ServiceSpec{}, errors.New("runtime port is invalid")
	}
	if !serviceModelID.MatchString(options.ModelID) {
		return ServiceSpec{}, errors.New("model id is invalid")
	}
	arguments := []string{
		"--model", modelPath,
		"--host", options.Host,
		"--port", strconv.Itoa(options.Port),
		"--model-id", options.ModelID,
	}
	return ServiceSpec{
		installRoot: root,
		executable:  executable,
		modelPath:   modelPath,
		host:        options.Host,
		port:        options.Port,
		modelID:     options.ModelID,
		arguments:   arguments,
		stdoutPath:  filepath.Join(root, "logs", "runtime.log"),
		stderrPath:  filepath.Join(root, "logs", "runtime-error.log"),
	}, nil
}

func (spec ServiceSpec) InstallRoot() string { return spec.installRoot }
func (spec ServiceSpec) Executable() string  { return spec.executable }
func (spec ServiceSpec) StdoutPath() string  { return spec.stdoutPath }
func (spec ServiceSpec) StderrPath() string  { return spec.stderrPath }
func (spec ServiceSpec) Arguments() []string { return append([]string(nil), spec.arguments...) }
func (spec ServiceSpec) HealthURL() string {
	return fmt.Sprintf("http://%s:%d/v1/models", spec.host, spec.port)
}

func RenderLaunchdUserPlist(spec ServiceSpec) ([]byte, error) {
	values := append([]string{spec.Executable()}, spec.Arguments()...)
	var arguments strings.Builder
	for _, value := range values {
		arguments.WriteString("    <string>")
		arguments.WriteString(escapeXML(value))
		arguments.WriteString("</string>\n")
	}
	payload := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>ai.metaflora.incubus</string>
  <key>ProgramArguments</key>
  <array>
%s  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
</dict>
</plist>
`, arguments.String(), escapeXML(spec.StdoutPath()), escapeXML(spec.StderrPath()))
	var parsed any
	if err := xml.Unmarshal([]byte(payload), &parsed); err != nil {
		return nil, err
	}
	return []byte(payload), nil
}

func RenderSystemdUserUnit(spec ServiceSpec) ([]byte, error) {
	parts := append([]string{quoteSystemd(spec.Executable())}, quoteSystemdArgs(spec.Arguments())...)
	payload := fmt.Sprintf(`[Unit]
Description=Metaflora Incubus v1
After=default.target

[Service]
Type=simple
ExecStart=%s
Restart=on-failure
RestartSec=2
StandardOutput=append:%s
StandardError=append:%s

[Install]
WantedBy=default.target
`, strings.Join(parts, " "), quoteSystemd(spec.StdoutPath()), quoteSystemd(spec.StderrPath()))
	return []byte(payload), nil
}

type ServiceCommand struct {
	Name string
	Args []string
}

type ServiceLifecycle struct {
	Start ServiceCommand
	Stop  ServiceCommand
}

func BuildServiceLifecycle(platform Platform, spec ServiceSpec) (ServiceLifecycle, error) {
	normalized, err := NormalizePlatform(platform.OS, platform.Arch)
	if err != nil {
		return ServiceLifecycle{}, err
	}
	switch normalized.OS {
	case "darwin":
		uid := "current-user"
		if current, lookupErr := user.Current(); lookupErr == nil && current.Uid != "" {
			uid = current.Uid
		}
		servicePath := filepath.Join(spec.InstallRoot(), "config", "ai.metaflora.incubus.plist")
		domain := "gui/" + uid
		return ServiceLifecycle{
			Start: ServiceCommand{"launchctl", []string{"bootstrap", domain, servicePath}},
			Stop:  ServiceCommand{"launchctl", []string{"bootout", domain, servicePath}},
		}, nil
	case "linux":
		return ServiceLifecycle{
			Start: ServiceCommand{"systemctl", []string{"--user", "enable", "--now", "metaflora-incubus.service"}},
			Stop:  ServiceCommand{"systemctl", []string{"--user", "disable", "--now", "metaflora-incubus.service"}},
		}, nil
	default:
		return ServiceLifecycle{}, errors.New("unsupported service platform")
	}
}

func WriteServiceFile(path string, payload []byte) error {
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return errors.New("refusing to replace a symlinked service file")
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".incubus-service-")
	if err != nil {
		return err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(payload); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}

func withinRoot(path, root string) bool {
	if !filepath.IsAbs(path) {
		return false
	}
	relative, err := filepath.Rel(root, path)
	return err == nil && relative != "." && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func escapeXML(value string) string {
	var output bytes.Buffer
	_ = xml.EscapeText(&output, []byte(value))
	return output.String()
}

func quoteSystemd(value string) string {
	escaped := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "%", "%%", "\n", `\n`).Replace(value)
	return `"` + escaped + `"`
}

func quoteSystemdArgs(values []string) []string {
	result := make([]string, len(values))
	for index, value := range values {
		result[index] = quoteSystemd(value)
	}
	return result
}
