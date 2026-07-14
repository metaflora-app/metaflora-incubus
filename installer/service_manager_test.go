package installer

import (
	"encoding/xml"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func validServiceSpec(t *testing.T) ServiceSpec {
	t.Helper()

	root := filepath.Join(t.TempDir(), "Metaflora Incubus")
	spec, err := NewServiceSpec(ServiceSpecOptions{
		InstallRoot: filepath.Clean(root),
		Executable:  filepath.Join(root, "current", "bin", "incubus-runtime"),
		ModelPath:   filepath.Join(root, "current", "models", "incubus-v1.gguf"),
		Host:        "127.0.0.1",
		Port:        18991,
		ModelID:     "metaflora-incubus-v1",
	})
	if err != nil {
		t.Fatalf("NewServiceSpec() error = %v", err)
	}
	return spec
}

func TestServiceSpecIsImmutableAndUsesArgumentVector(t *testing.T) {
	spec := validServiceSpec(t)
	typeOfSpec := reflect.TypeOf(spec)
	for index := 0; index < typeOfSpec.NumField(); index++ {
		if typeOfSpec.Field(index).PkgPath == "" {
			t.Fatalf("ServiceSpec field %q is exported and therefore mutable", typeOfSpec.Field(index).Name)
		}
	}

	first := spec.Arguments()
	first[0] = "--compromised"
	second := spec.Arguments()
	if second[0] == "--compromised" {
		t.Fatal("Arguments() exposed mutable internal state")
	}
	joined := strings.Join(second, " ")
	for _, required := range []string{"--host 127.0.0.1", "--port 18991", "--model-id metaflora-incubus-v1"} {
		if !strings.Contains(joined, required) {
			t.Fatalf("runtime arguments %q do not contain %q", joined, required)
		}
	}
	if spec.StdoutPath() != filepath.Join(spec.InstallRoot(), "logs", "runtime.log") {
		t.Fatalf("StdoutPath() = %q", spec.StdoutPath())
	}
	if spec.StderrPath() != filepath.Join(spec.InstallRoot(), "logs", "runtime-error.log") {
		t.Fatalf("StderrPath() = %q", spec.StderrPath())
	}
}

func TestNewServiceSpecRejectsUnsafeRuntimeConfiguration(t *testing.T) {
	root := t.TempDir()
	tests := []struct {
		name    string
		options ServiceSpecOptions
	}{
		{
			name: "non-loopback wildcard bind",
			options: ServiceSpecOptions{
				InstallRoot: root, Executable: filepath.Join(root, "runtime"),
				ModelPath: filepath.Join(root, "model.gguf"), Host: "0.0.0.0", Port: 18991,
				ModelID: "metaflora-incubus-v1",
			},
		},
		{
			name: "IPv6 wildcard bind",
			options: ServiceSpecOptions{
				InstallRoot: root, Executable: filepath.Join(root, "runtime"),
				ModelPath: filepath.Join(root, "model.gguf"), Host: "::", Port: 18991,
				ModelID: "metaflora-incubus-v1",
			},
		},
		{
			name: "executable outside managed root",
			options: ServiceSpecOptions{
				InstallRoot: root, Executable: filepath.Join(filepath.Dir(root), "foreign-runtime"),
				ModelPath: filepath.Join(root, "model.gguf"), Host: "127.0.0.1", Port: 18991,
				ModelID: "metaflora-incubus-v1",
			},
		},
		{
			name: "model outside managed root",
			options: ServiceSpecOptions{
				InstallRoot: root, Executable: filepath.Join(root, "runtime"),
				ModelPath: filepath.Join(filepath.Dir(root), "foreign-model.gguf"), Host: "127.0.0.1", Port: 18991,
				ModelID: "metaflora-incubus-v1",
			},
		},
		{
			name: "newline injection in model id",
			options: ServiceSpecOptions{
				InstallRoot: root, Executable: filepath.Join(root, "runtime"),
				ModelPath: filepath.Join(root, "model.gguf"), Host: "127.0.0.1", Port: 18991,
				ModelID: "metaflora-incubus-v1\nRun=evil",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewServiceSpec(test.options); err == nil {
				t.Fatal("NewServiceSpec() accepted unsafe input")
			}
		})
	}
}

func TestRenderLaunchdUserPlistUsesProgramArgumentsAndLeastPrivilegePaths(t *testing.T) {
	spec := validServiceSpec(t)
	payload, err := RenderLaunchdUserPlist(spec)
	if err != nil {
		t.Fatalf("RenderLaunchdUserPlist() error = %v", err)
	}
	text := string(payload)
	for _, forbidden := range []string{"/bin/sh", "sh -c", "sudo", "UserName", "Program</key>"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("launchd plist contains unsafe or system-level token %q", forbidden)
		}
	}
	for _, required := range []string{
		"<key>ProgramArguments</key>", "<key>RunAtLoad</key>", "<key>KeepAlive</key>",
	} {
		if !strings.Contains(text, required) {
			t.Fatalf("launchd plist does not contain %q", required)
		}
	}
	for _, required := range append([]string{
		spec.Executable(), spec.StdoutPath(), spec.StderrPath(),
	}, spec.Arguments()...) {
		if !strings.Contains(text, xmlEscape(required)) {
			t.Fatalf("launchd plist does not contain %q", required)
		}
	}
	var document any
	if err := xml.Unmarshal(payload, &document); err != nil {
		t.Fatalf("launchd plist is invalid XML: %v", err)
	}
}

func TestRenderSystemdUserUnitUsesDirectExecAndInstallRootLogs(t *testing.T) {
	spec := validServiceSpec(t)
	payload, err := RenderSystemdUserUnit(spec)
	if err != nil {
		t.Fatalf("RenderSystemdUserUnit() error = %v", err)
	}
	text := string(payload)
	for _, forbidden := range []string{"/bin/sh", "sh -c", "sudo", "User=", "0.0.0.0"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("systemd user unit contains unsafe or system-level token %q", forbidden)
		}
	}
	for _, required := range []string{
		"ExecStart=", systemdQuote(spec.Executable()),
		"StandardOutput=append:" + systemdQuote(spec.StdoutPath()),
		"StandardError=append:" + systemdQuote(spec.StderrPath()),
		"WantedBy=default.target", "Restart=on-failure",
	} {
		if !strings.Contains(text, required) {
			t.Fatalf("systemd user unit does not contain %q\n%s", required, text)
		}
	}
}

func TestRenderSystemdUserUnitEscapesSpecifierCharacters(t *testing.T) {
	root := filepath.Join(t.TempDir(), "Incubus %h")
	spec, err := NewServiceSpec(ServiceSpecOptions{
		InstallRoot: root, Executable: filepath.Join(root, "runtime"),
		ModelPath: filepath.Join(root, "model.gguf"), Host: "127.0.0.1", Port: 18991,
		ModelID: "metaflora-incubus-v1",
	})
	if err != nil {
		t.Fatal(err)
	}
	payload, err := RenderSystemdUserUnit(spec)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(payload), "Incubus %h") || !strings.Contains(string(payload), "Incubus %%h") {
		t.Fatalf("systemd specifier was not escaped:\n%s", payload)
	}
}

func TestBuildServiceLifecycleUsesPerUserCommandsAndSeparateArguments(t *testing.T) {
	spec := validServiceSpec(t)
	tests := []struct {
		platform Platform
		start    string
		stop     string
		marker   string
	}{
		{platform: Platform{OS: "darwin", Arch: "arm64"}, start: "launchctl", stop: "launchctl", marker: "gui/"},
		{platform: Platform{OS: "linux", Arch: "amd64"}, start: "systemctl", stop: "systemctl", marker: "--user"},
	}

	for _, test := range tests {
		t.Run(test.platform.OS, func(t *testing.T) {
			lifecycle, err := BuildServiceLifecycle(test.platform, spec)
			if err != nil {
				t.Fatalf("BuildServiceLifecycle() error = %v", err)
			}
			if lifecycle.Start.Name != test.start || lifecycle.Stop.Name != test.stop {
				t.Fatalf("unexpected lifecycle commands: %#v", lifecycle)
			}
			joined := strings.Join(append(append([]string{}, lifecycle.Start.Args...), lifecycle.Stop.Args...), " ")
			if !strings.Contains(joined, test.marker) {
				t.Fatalf("per-user marker %q absent from %q", test.marker, joined)
			}
			if strings.Contains(joined, "sudo") || strings.Contains(joined, "sh -c") {
				t.Fatalf("lifecycle escalates privilege or invokes a shell: %q", joined)
			}
		})
	}
}

func TestServiceHealthURLIsLoopbackOnly(t *testing.T) {
	spec := validServiceSpec(t)
	if got := spec.HealthURL(); got != "http://127.0.0.1:18991/v1/models" {
		t.Fatalf("HealthURL() = %q", got)
	}
}

func TestWriteServiceFileCreatesPrivateFileAndRefusesSymlinkTarget(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "user services", "incubus.service")
	if err := WriteServiceFile(target, []byte("private service definition")); err != nil {
		t.Fatalf("WriteServiceFile() error = %v", err)
	}
	info, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("service file mode = %o, want 600", info.Mode().Perm())
	}

	foreign := filepath.Join(root, "foreign")
	if err := os.WriteFile(foreign, []byte("must survive"), 0o600); err != nil {
		t.Fatal(err)
	}
	symlink := filepath.Join(root, "incubus.plist")
	if err := os.Symlink(foreign, symlink); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("symlink unavailable: %v", err)
		}
		t.Fatal(err)
	}
	if err := WriteServiceFile(symlink, []byte("overwritten")); err == nil {
		t.Fatal("WriteServiceFile() followed an existing symlink")
	}
	payload, err := os.ReadFile(foreign)
	if err != nil {
		t.Fatal(err)
	}
	if string(payload) != "must survive" {
		t.Fatalf("symlink target was modified: %q", payload)
	}
}

func xmlEscape(value string) string {
	value = strings.ReplaceAll(value, "&", "&amp;")
	value = strings.ReplaceAll(value, "<", "&lt;")
	return strings.ReplaceAll(value, ">", "&gt;")
}

func systemdQuote(value string) string {
	return `"` + strings.ReplaceAll(value, `"`, `\"`) + `"`
}
