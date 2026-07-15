package installer

import (
	"context"
	"crypto/ed25519"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	defaultManifestURL  = "https://github.com/metaflora-app/metaflora-incubus/releases/latest/download/manifest.json"
	defaultSignatureURL = "https://github.com/metaflora-app/metaflora-incubus/releases/latest/download/manifest.json.sig"
	defaultRuntimePort  = 18991
)

type ProductionOptions struct {
	PinnedPublicKey ed25519.PublicKey
	ManifestURL     string
	SignatureURL    string
	Executable      string
}

func NewProductionCLIConfig(options ProductionOptions) (CLIConfig, error) {
	if len(options.PinnedPublicKey) != ed25519.PublicKeySize {
		return CLIConfig{}, errors.New("release signing public key is not embedded in this incubusctl build")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return CLIConfig{}, err
	}
	executable := options.Executable
	if executable == "" {
		executable, err = os.Executable()
		if err != nil {
			return CLIConfig{}, err
		}
	}
	executable, err = filepath.Abs(executable)
	if err != nil {
		return CLIConfig{}, err
	}
	root, err := defaultInstallRoot(home)
	if err != nil {
		return CLIConfig{}, err
	}
	manifestURL := options.ManifestURL
	if manifestURL == "" {
		manifestURL = defaultManifestURL
	}
	signatureURL := options.SignatureURL
	if signatureURL == "" {
		signatureURL = defaultSignatureURL
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = 45 * time.Second
	client := &http.Client{Transport: transport}
	platform, err := NormalizePlatform(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return CLIConfig{}, err
	}
	service := nativeUserService{root: root, platform: platform, client: client, home: home, runner: ExecServiceCommandRunner{}}
	return CLIConfig{
		ManifestURL: manifestURL, ManifestSignatureURL: signatureURL,
		PinnedPublicKey: options.PinnedPublicKey, HTTPClient: client, Platform: platform,
		ProbeResources: probeHostResources, InstallRoot: root,
		OpenCodeConfigPath: filepath.Join(home, ".config", "opencode", "opencode.json"),
		EnsureRuntime:      service.ensure, StopRuntime: service.stop, CleanupRuntime: service.cleanup,
		RegisterOllama: registerOllama, UnregisterOllama: unregisterOllama,
		ControlBinarySource:      executable,
		ControlBinaryDestination: filepath.Join(home, ".local", "bin", controlBinaryName(platform.OS)),
		ValidateArtifact:         ValidateHostedArtifact,
	}, nil
}

func defaultInstallRoot(home string) (string, error) {
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "Metaflora Incubus"), nil
	case "linux":
		if dataHome := os.Getenv("XDG_DATA_HOME"); dataHome != "" && filepath.IsAbs(dataHome) {
			return filepath.Join(dataHome, "metaflora-incubus"), nil
		}
		return filepath.Join(home, ".local", "share", "metaflora-incubus"), nil
	case "windows":
		if local := os.Getenv("LOCALAPPDATA"); local != "" && filepath.IsAbs(local) {
			return filepath.Join(local, "Metaflora Incubus"), nil
		}
		return "", errors.New("LOCALAPPDATA is unavailable")
	default:
		return "", errors.New("unsupported operating system")
	}
}

type nativeUserService struct {
	root     string
	platform Platform
	client   *http.Client
	home     string
	runner   ExecServiceCommandRunner
}

func (service nativeUserService) spec(runtimeSpec RuntimeSpec) (ServiceSpec, error) {
	return NewServiceSpec(ServiceSpecOptions{
		InstallRoot: service.root,
		Executable:  filepath.Join(service.root, "current", "bin", runtimeBinaryName(service.platform.OS)),
		ModelPath:   runtimeSpec.ModelPath, Host: runtimeSpec.Host, Port: defaultRuntimePort,
		ModelID: runtimeSpec.ModelID,
	})
}

func (service nativeUserService) ensure(ctx context.Context, runtimeSpec RuntimeSpec) (RuntimeEndpoint, error) {
	spec, err := service.spec(runtimeSpec)
	if err != nil {
		return RuntimeEndpoint{}, err
	}
	endpoint := RuntimeEndpoint{BaseURL: fmt.Sprintf("http://127.0.0.1:%d/v1", defaultRuntimePort)}
	definitionPath, err := service.writeDefinition(spec)
	if err != nil {
		return RuntimeEndpoint{}, err
	}
	lifecycle, err := BuildServiceLifecycle(service.platform, spec)
	if err != nil {
		return RuntimeEndpoint{}, err
	}
	lifecycle = service.lifecycleAtPath(lifecycle, definitionPath)
	if service.platform.OS == "linux" {
		if err := service.runner.Run(ctx, ServiceCommand{Name: "systemctl", Args: []string{"--user", "daemon-reload"}}); err != nil {
			return RuntimeEndpoint{}, err
		}
	}
	_ = service.runner.Run(ctx, lifecycle.Stop)
	if err := service.runner.Run(ctx, lifecycle.Start); err != nil {
		return RuntimeEndpoint{}, err
	}
	if err := waitForHealth(ctx, service.client, endpoint.BaseURL, 30*time.Second); err != nil {
		_ = definitionPath
		return RuntimeEndpoint{}, err
	}
	return endpoint, nil
}

func (service nativeUserService) stop(ctx context.Context, runtimeSpec RuntimeSpec) error {
	spec, err := service.spec(runtimeSpec)
	if err != nil {
		return err
	}
	lifecycle, err := BuildServiceLifecycle(service.platform, spec)
	if err != nil {
		return err
	}
	definitionPath, pathErr := service.definitionPath(spec)
	if pathErr != nil {
		return pathErr
	}
	lifecycle = service.lifecycleAtPath(lifecycle, definitionPath)
	err = service.runner.Run(ctx, lifecycle.Stop)
	if service.platform.OS == "darwin" && isMissingLaunchdService(err) {
		return nil
	}
	return err
}

func isMissingLaunchdService(err error) bool {
	if err == nil {
		return false
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "could not find service") || strings.Contains(message, "no such process") || strings.Contains(message, "service not found")
}

func (service nativeUserService) lifecycleAtPath(lifecycle ServiceLifecycle, definitionPath string) ServiceLifecycle {
	if service.platform.OS != "darwin" {
		return lifecycle
	}
	startArgs := append([]string(nil), lifecycle.Start.Args...)
	stopArgs := append([]string(nil), lifecycle.Stop.Args...)
	if len(startArgs) > 0 {
		startArgs[len(startArgs)-1] = definitionPath
	}
	if len(stopArgs) > 0 {
		stopArgs[len(stopArgs)-1] = definitionPath
	}
	return ServiceLifecycle{
		Start: ServiceCommand{Name: lifecycle.Start.Name, Args: startArgs},
		Stop:  ServiceCommand{Name: lifecycle.Stop.Name, Args: stopArgs},
	}
}

func (service nativeUserService) cleanup(ctx context.Context, runtimeSpec RuntimeSpec) error {
	spec, err := service.spec(runtimeSpec)
	if err != nil {
		return err
	}
	path, err := service.definitionPath(spec)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	if service.platform.OS == "linux" {
		return service.runner.Run(ctx, ServiceCommand{Name: "systemctl", Args: []string{"--user", "daemon-reload"}})
	}
	return nil
}

func (service nativeUserService) writeDefinition(spec ServiceSpec) (string, error) {
	path, err := service.definitionPath(spec)
	if err != nil {
		return "", err
	}
	var payload []byte
	switch service.platform.OS {
	case "darwin":
		payload, err = RenderLaunchdUserPlist(spec)
	case "linux":
		payload, err = RenderSystemdUserUnit(spec)
	default:
		return "", errors.New("persistent user service is unsupported on this operating system")
	}
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Join(service.root, "logs"), 0o700); err != nil {
		return "", err
	}
	return path, WriteServiceFile(path, payload)
}

func (service nativeUserService) definitionPath(spec ServiceSpec) (string, error) {
	switch service.platform.OS {
	case "darwin":
		return filepath.Join(service.home, "Library", "LaunchAgents", "ai.metaflora.incubus.plist"), nil
	case "linux":
		return filepath.Join(service.home, ".config", "systemd", "user", "metaflora-incubus.service"), nil
	default:
		return "", errors.New("persistent user service is unsupported on this operating system")
	}
}

func waitForHealth(ctx context.Context, client *http.Client, baseURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		attempt, cancel := context.WithTimeout(ctx, time.Second)
		err := checkRuntimeHealth(attempt, client, baseURL)
		cancel()
		if err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("runtime did not become healthy: %w", err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func probeHostResources(ctx context.Context, path string) (Resources, error) {
	ram, err := probeRAM(ctx)
	if err != nil {
		return Resources{}, err
	}
	probePath := path
	for {
		if _, err := os.Stat(probePath); err == nil {
			break
		}
		parent := filepath.Dir(probePath)
		if parent == probePath {
			return Resources{}, errors.New("cannot locate filesystem for install root")
		}
		probePath = parent
	}
	if runtime.GOOS == "windows" {
		return Resources{}, errors.New("automatic Windows disk preflight is unsupported in this release")
	}
	output, err := exec.CommandContext(ctx, "df", "-Pk", probePath).Output()
	if err != nil {
		return Resources{}, fmt.Errorf("disk preflight: %w", err)
	}
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) < 2 {
		return Resources{}, errors.New("unexpected disk preflight output")
	}
	fields := strings.Fields(lines[len(lines)-1])
	if len(fields) < 4 {
		return Resources{}, errors.New("unexpected disk preflight fields")
	}
	availableKiB, err := strconv.ParseUint(fields[3], 10, 64)
	if err != nil {
		return Resources{}, err
	}
	return Resources{RAMBytes: ram, FreeDiskBytes: availableKiB * 1024}, nil
}

func probeRAM(ctx context.Context) (uint64, error) {
	switch runtime.GOOS {
	case "darwin":
		output, err := exec.CommandContext(ctx, "sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			return 0, err
		}
		return strconv.ParseUint(strings.TrimSpace(string(output)), 10, 64)
	case "linux":
		payload, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			return 0, err
		}
		for _, line := range strings.Split(string(payload), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= 2 && fields[0] == "MemTotal:" {
				value, err := strconv.ParseUint(fields[1], 10, 64)
				return value * 1024, err
			}
		}
		return 0, errors.New("MemTotal is absent from /proc/meminfo")
	default:
		return 0, errors.New("automatic RAM preflight is unsupported on this operating system")
	}
}

func registerOllama(ctx context.Context, profile OllamaProfile) error {
	file, err := os.CreateTemp("", "incubus-Modelfile-")
	if err != nil {
		return err
	}
	path := file.Name()
	defer os.Remove(path)
	if err := file.Chmod(0o600); err != nil {
		file.Close()
		return err
	}
	if _, err := file.WriteString(profile.Modelfile); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	output, err := exec.CommandContext(ctx, "ollama", "create", profile.Name, "-f", path).CombinedOutput()
	if err != nil {
		return fmt.Errorf("ollama create failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

func unregisterOllama(ctx context.Context, profile OllamaProfile) error {
	output, err := exec.CommandContext(ctx, "ollama", "rm", profile.Name).CombinedOutput()
	if err != nil {
		return fmt.Errorf("ollama rm failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}
