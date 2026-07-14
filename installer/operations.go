package installer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const maximumLogBytes = 1024 * 1024

func controlBinaryName(osName string) string {
	if strings.EqualFold(strings.TrimSpace(osName), "windows") {
		return "incubusctl.exe"
	}
	return "incubusctl"
}

func persistControlBinary(source, destination string) (string, bool, bool, error) {
	sourceInfo, err := os.Stat(source)
	if err != nil {
		return "", false, false, err
	}
	if !sourceInfo.Mode().IsRegular() {
		return "", false, false, errors.New("control binary source is not a regular file")
	}
	if destinationInfo, statErr := os.Stat(destination); statErr == nil && os.SameFile(sourceInfo, destinationInfo) {
		return "", false, false, nil
	}
	if existing, lstatErr := os.Lstat(destination); lstatErr == nil && existing.Mode()&os.ModeSymlink != 0 {
		return "", false, false, errors.New("refusing to replace a symlinked control binary")
	} else if lstatErr != nil && !os.IsNotExist(lstatErr) {
		return "", false, false, lstatErr
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
		return "", false, false, err
	}
	input, err := os.Open(source)
	if err != nil {
		return "", false, false, err
	}
	defer input.Close()
	temporary, err := os.CreateTemp(filepath.Dir(destination), ".incubusctl-")
	if err != nil {
		return "", false, false, err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if err := temporary.Chmod(0o700); err != nil {
		temporary.Close()
		return "", false, false, err
	}
	if _, err := io.Copy(temporary, input); err != nil {
		temporary.Close()
		return "", false, false, err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return "", false, false, err
	}
	if err := temporary.Close(); err != nil {
		return "", false, false, err
	}
	backup := destination + ".rollback"
	_ = os.Remove(backup)
	hadPrevious := false
	if _, err := os.Stat(destination); err == nil {
		hadPrevious = true
		if err := os.Rename(destination, backup); err != nil {
			return "", false, false, err
		}
	} else if !os.IsNotExist(err) {
		return "", false, false, err
	}
	if err := os.Rename(temporaryName, destination); err != nil {
		if hadPrevious {
			_ = os.Rename(backup, destination)
		}
		return "", false, false, err
	}
	return backup, hadPrevious, true, nil
}

func installedState(config CLIConfig) (installState, string, error) {
	path := filepath.Join(config.InstallRoot, "install-state.json")
	state, err := readInstallState(path)
	if os.IsNotExist(err) {
		return installState{}, path, errors.New("Metaflora Incubus v1 is not installed")
	}
	return state, path, err
}

func startInstalled(ctx context.Context, config CLIConfig) error {
	state, statePath, err := installedState(config)
	if err != nil {
		return err
	}
	endpoint, err := config.EnsureRuntime(ctx, state.Runtime)
	if err != nil {
		return err
	}
	state.RuntimeBaseURL = endpoint.BaseURL
	return writeJSON(statePath, state)
}

func stopInstalled(ctx context.Context, config CLIConfig) error {
	state, _, err := installedState(config)
	if err != nil {
		return err
	}
	return config.StopRuntime(ctx, state.Runtime)
}

func restartInstalled(ctx context.Context, config CLIConfig) error {
	if err := stopInstalled(ctx, config); err != nil {
		return err
	}
	return startInstalled(ctx, config)
}

func statusInstalled(ctx context.Context, output io.Writer, config CLIConfig) error {
	state, _, err := installedState(config)
	if err != nil {
		return err
	}
	if err := checkRuntimeHealth(ctx, config.HTTPClient, state.RuntimeBaseURL); err != nil {
		return fmt.Errorf("installed but not running: %w", err)
	}
	_, err = fmt.Fprintf(output, "running %s (%s)\n", state.Runtime.ModelID, state.Release)
	return err
}

func doctorInstalled(ctx context.Context, output io.Writer, config CLIConfig) error {
	state, _, err := installedState(config)
	if err != nil {
		return err
	}
	for _, path := range []string{state.Runtime.ModelPath, filepath.Join(config.InstallRoot, "current", "bin", runtimeBinaryName(config.Platform.OS))} {
		info, statErr := os.Stat(path)
		if statErr != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("required runtime file is missing: %s", path)
		}
	}
	if err := checkRuntimeHealth(ctx, config.HTTPClient, state.RuntimeBaseURL); err != nil {
		return fmt.Errorf("runtime health check failed: %w", err)
	}
	_, err = fmt.Fprintln(output, "doctor: installation and runtime are healthy")
	return err
}

func runtimeBinaryName(osName string) string {
	if strings.EqualFold(strings.TrimSpace(osName), "windows") {
		return "incubus-runtime.exe"
	}
	return "incubus-runtime"
}

func checkRuntimeHealth(ctx context.Context, client *http.Client, baseURL string) error {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Scheme != "http" || parsed.Hostname() != "127.0.0.1" || parsed.Port() == "" {
		return errors.New("stored runtime endpoint is not loopback HTTP")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/models", nil)
	if err != nil {
		return err
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", response.StatusCode)
	}
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	decoder := json.NewDecoder(io.LimitReader(response.Body, 1024*1024))
	if err := decoder.Decode(&models); err != nil {
		return fmt.Errorf("invalid models response: %w", err)
	}
	for _, model := range models.Data {
		if model.ID == productModelID {
			return nil
		}
	}
	return errors.New("runtime does not report the Incubus model id")
}

func printLogs(output io.Writer, config CLIConfig) error {
	if _, _, err := installedState(config); err != nil {
		return err
	}
	found := false
	for _, name := range []string{"runtime.log", "runtime-error.log"} {
		path := filepath.Join(config.InstallRoot, "logs", name)
		file, err := os.Open(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		found = true
		_, copyErr := io.Copy(output, io.LimitReader(file, maximumLogBytes))
		closeErr := file.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	if !found {
		return errors.New("runtime logs do not exist yet")
	}
	return nil
}

func integrateInstalled(ctx context.Context, args []string, config CLIConfig) error {
	if len(args) != 1 {
		return errors.New("usage: incubusctl integrate opencode|ollama")
	}
	state, statePath, err := installedState(config)
	if err != nil {
		return err
	}
	switch strings.ToLower(args[0]) {
	case "opencode":
		if state.RuntimeBaseURL == "" {
			return errors.New("runtime endpoint is missing from install state")
		}
		_, installed, err := mergeOpenCode(config.OpenCodeConfigPath, state.RuntimeBaseURL)
		if err != nil {
			return err
		}
		state.InstalledProvider = installed
		return writeJSON(statePath, state)
	case "ollama":
		if config.RegisterOllama == nil {
			return errors.New("Ollama integration is unsupported on this host")
		}
		modelfile, err := RenderOllamaModelfile(state.Runtime.ModelPath)
		if err != nil {
			return err
		}
		profile := OllamaProfile{Name: productModelID, Modelfile: string(modelfile)}
		if err := config.RegisterOllama(ctx, profile); err != nil {
			return err
		}
		state.Ollama = &profile
		return writeJSON(statePath, state)
	default:
		return fmt.Errorf("unsupported integration %q", args[0])
	}
}

type ExecServiceCommandRunner struct{}

func (ExecServiceCommandRunner) Run(ctx context.Context, command ServiceCommand) error {
	if strings.TrimSpace(command.Name) == "" || strings.ContainsRune(command.Name, '\x00') {
		return errors.New("service command executable is invalid")
	}
	for _, argument := range command.Args {
		if strings.ContainsRune(argument, '\x00') {
			return errors.New("service command argument contains NUL")
		}
	}
	process := exec.CommandContext(ctx, command.Name, command.Args...)
	output, err := process.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s failed: %w: %s", command.Name, err, strings.TrimSpace(string(output)))
	}
	return nil
}
