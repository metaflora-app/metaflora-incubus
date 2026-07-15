package installer

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	productProviderID         = "metaflora-incubus"
	productModelID            = "metaflora-incubus-v1"
	artifactInactivityTimeout = 45 * time.Second
)

type RuntimeSpec struct {
	Host      string `json:"host"`
	ModelPath string `json:"model_path"`
	ModelID   string `json:"model_id"`
}

type RuntimeEndpoint struct {
	BaseURL string
}

type OllamaProfile struct {
	Name      string `json:"name"`
	Modelfile string `json:"modelfile"`
}

type CLIConfig struct {
	ManifestURL              string
	ManifestSignatureURL     string
	PinnedPublicKey          ed25519.PublicKey
	HTTPClient               *http.Client
	Platform                 Platform
	ProbeResources           func(context.Context, string) (Resources, error)
	InstallRoot              string
	OpenCodeConfigPath       string
	EnsureRuntime            func(context.Context, RuntimeSpec) (RuntimeEndpoint, error)
	StopRuntime              func(context.Context, RuntimeSpec) error
	CleanupRuntime           func(context.Context, RuntimeSpec) error
	RegisterOllama           func(context.Context, OllamaProfile) error
	UnregisterOllama         func(context.Context, OllamaProfile) error
	BeforeActivate           func(stagingDir, targetDir string) error
	ControlBinarySource      string
	ControlBinaryDestination string
	ValidateArtifact         func(Artifact) error
}

type installState struct {
	SchemaVersion     int             `json:"schema_version"`
	Release           string          `json:"release"`
	ArtifactID        string          `json:"artifact_id"`
	Runtime           RuntimeSpec     `json:"runtime"`
	PreviousProvider  json.RawMessage `json:"previous_provider,omitempty"`
	InstalledProvider json.RawMessage `json:"installed_provider"`
	Ollama            *OllamaProfile  `json:"ollama,omitempty"`
	RuntimeBaseURL    string          `json:"runtime_base_url,omitempty"`
	ControlBinary     string          `json:"control_binary,omitempty"`
	ModelSHA256       string          `json:"model_sha256"`
	ModelSizeBytes    uint64          `json:"model_size_bytes"`
	RuntimeSHA256     string          `json:"runtime_sha256"`
	RuntimeSizeBytes  uint64          `json:"runtime_size_bytes"`
	ServerSHA256      string          `json:"server_sha256"`
	ServerSizeBytes   uint64          `json:"server_size_bytes"`
}

func RunCLI(ctx context.Context, args []string, stdout, stderr io.Writer, config CLIConfig) int {
	if err := validateCLIConfig(config); err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 2
	}
	if len(args) == 0 {
		_, _ = fmt.Fprintln(stderr, "usage: incubusctl install|uninstall|status|start|stop|restart|logs|doctor|update|integrate")
		return 2
	}
	var err error
	switch args[0] {
	case "install":
		err = install(ctx, config, containsArg(args[1:], "--register-ollama"))
	case "uninstall":
		err = uninstall(ctx, config)
	case "start":
		err = startInstalled(ctx, config)
	case "stop":
		err = stopInstalled(ctx, config)
	case "restart":
		err = restartInstalled(ctx, config)
	case "status":
		err = statusInstalled(ctx, stdout, config)
	case "logs":
		err = printLogs(stdout, config)
	case "doctor":
		err = doctorInstalled(ctx, stdout, config)
	case "update":
		err = install(ctx, config, false)
	case "integrate":
		err = integrateInstalled(ctx, args[1:], config)
	case "rollback", "profile", "bundle", "move":
		err = fmt.Errorf("unsupported command %q in this release", args[0])
	default:
		err = fmt.Errorf("unknown command %q", args[0])
	}
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 1
	}
	if args[0] != "status" && args[0] != "logs" && args[0] != "doctor" {
		_, _ = fmt.Fprintln(stdout, "Metaflora Incubus v1 ready")
	}
	return 0
}

func validateCLIConfig(config CLIConfig) error {
	if config.HTTPClient == nil || len(config.PinnedPublicKey) != ed25519.PublicKeySize {
		return errors.New("installer trust configuration is incomplete")
	}
	if config.ProbeResources == nil || config.EnsureRuntime == nil || config.StopRuntime == nil {
		return errors.New("installer runtime configuration is incomplete")
	}
	if !filepath.IsAbs(config.InstallRoot) || !filepath.IsAbs(config.OpenCodeConfigPath) {
		return errors.New("installer paths must be absolute")
	}
	if config.ControlBinarySource != "" && !filepath.IsAbs(config.ControlBinarySource) {
		return errors.New("control binary source must be absolute")
	}
	if config.ControlBinaryDestination != "" && !filepath.IsAbs(config.ControlBinaryDestination) {
		return errors.New("control binary destination must be absolute")
	}
	return nil
}

func install(ctx context.Context, config CLIConfig, registerOllama bool) (resultErr error) {
	manifestPayload, err := fetch(ctx, config.HTTPClient, config.ManifestURL)
	if err != nil {
		return err
	}
	manifestSignature, err := fetch(ctx, config.HTTPClient, config.ManifestSignatureURL)
	if err != nil {
		return err
	}
	manifest, err := ParseAndVerifyManifest(manifestPayload, manifestSignature, config.PinnedPublicKey)
	if err != nil {
		return err
	}
	resources, err := config.ProbeResources(ctx, config.InstallRoot)
	if err != nil {
		return fmt.Errorf("resource preflight: %w", err)
	}
	statePath := filepath.Join(config.InstallRoot, "install-state.json")
	previousState, previousStateError := readInstallState(statePath, config.InstallRoot)
	matchedArtifacts, err := SelectReleaseArtifacts(manifest, config.Platform, Resources{RAMBytes: resources.RAMBytes, FreeDiskBytes: ^uint64(0)})
	if err != nil {
		return err
	}
	if config.ValidateArtifact != nil {
		for _, artifact := range []Artifact{matchedArtifacts.Runtime, matchedArtifacts.Model} {
			if err := config.ValidateArtifact(artifact); err != nil {
				return fmt.Errorf("hosted artifact policy: %w", err)
			}
		}
	}
	releaseArtifactID := matchedArtifacts.Runtime.ID + "+" + matchedArtifacts.Model.ID
	if previousStateError == nil && previousState.Ollama != nil {
		registerOllama = true
	}
	if previousStateError == nil && previousState.Release == manifest.Release && previousState.ArtifactID == releaseArtifactID {
		if err := verifyInstalledFiles(previousState, config); err != nil {
			return fmt.Errorf("same-release integrity check failed; run incubusctl update or reinstall: %w", err)
		}
		_, err = config.EnsureRuntime(ctx, previousState.Runtime)
		return err
	}
	releaseArtifacts, err := SelectReleaseArtifacts(manifest, config.Platform, resources)
	if err != nil {
		if previousStateError == nil && strings.Contains(err.Error(), "disk") {
			return errors.New("safe update requires space for both old and new weights; run incubusctl uninstall, then install again to update on a low-disk machine")
		}
		return err
	}

	if err := os.MkdirAll(config.InstallRoot, 0o700); err != nil {
		return err
	}
	staging, err := os.MkdirTemp(config.InstallRoot, ".staging-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(staging)
	archivePath := filepath.Join(staging, "release.tar.gz")
	if err := downloadArtifact(
		ctx, config.HTTPClient, releaseArtifacts.Runtime, config.PinnedPublicKey, archivePath,
	); err != nil {
		return err
	}
	if err := extractTarGzip(archivePath, staging, releaseArtifacts.Runtime.UnpackedSizeBytes); err != nil {
		return err
	}
	if err := os.Remove(archivePath); err != nil {
		return err
	}
	if err := validateStagedRuntime(staging, config.Platform.OS); err != nil {
		return err
	}
	runtimeSHA256, runtimeSizeBytes, err := fileIntegrity(filepath.Join(staging, "bin", runtimeBinaryName(config.Platform.OS)))
	if err != nil {
		return fmt.Errorf("inspect runtime wrapper: %w", err)
	}
	serverSHA256, serverSizeBytes, err := fileIntegrity(filepath.Join(staging, "bin", "llama-server"))
	if err != nil {
		return fmt.Errorf("inspect inference server: %w", err)
	}
	modelDestination := filepath.Join(staging, "models", "incubus-v1.gguf")
	if err := os.MkdirAll(filepath.Dir(modelDestination), 0o700); err != nil {
		return err
	}
	if err := downloadArtifact(ctx, config.HTTPClient, releaseArtifacts.Model, config.PinnedPublicKey, modelDestination); err != nil {
		return fmt.Errorf("download direct GGUF: %w", err)
	}
	modelSHA256, modelSizeBytes, err := fileIntegrity(modelDestination)
	if err != nil {
		return fmt.Errorf("inspect hosted model: %w", err)
	}
	runtimeSpec := RuntimeSpec{
		Host:      "127.0.0.1",
		ModelPath: filepath.Join(config.InstallRoot, "current", "models", "incubus-v1.gguf"),
		ModelID:   productModelID,
	}
	if err := writeJSON(filepath.Join(staging, "config", "runtime.json"), runtimeSpec); err != nil {
		return err
	}
	current := filepath.Join(config.InstallRoot, "current")
	if config.BeforeActivate != nil {
		if err := config.BeforeActivate(staging, current); err != nil {
			return err
		}
	}
	backup, hadPrevious, err := activate(staging, current)
	if err != nil {
		return err
	}
	committed := false
	runtimeStarted := false
	providerChanged := false
	ollamaRegistered := false
	var providerBeforeUpdate json.RawMessage
	var installedProvider json.RawMessage
	var registeredProfile OllamaProfile
	controlDestination := ""
	controlBackup := ""
	controlHadPrevious := false
	controlChanged := false
	defer func() {
		if committed {
			return
		}
		var recoveryErrors []error
		if ollamaRegistered && config.UnregisterOllama != nil {
			if err := config.UnregisterOllama(ctx, registeredProfile); err != nil {
				recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback Ollama: %w", err))
			}
		}
		if providerChanged {
			if err := revertOpenCode(config.OpenCodeConfigPath, installState{
				InstalledProvider: installedProvider,
				PreviousProvider:  providerBeforeUpdate,
			}); err != nil {
				recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback OpenCode: %w", err))
			}
		}
		if runtimeStarted {
			if err := config.StopRuntime(ctx, runtimeSpec); err != nil {
				recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback stop runtime: %w", err))
			}
		}
		if previousStateError != nil && config.CleanupRuntime != nil {
			if err := config.CleanupRuntime(ctx, runtimeSpec); err != nil {
				recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback cleanup runtime: %w", err))
			}
		}
		if err := os.RemoveAll(current); err != nil {
			recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback remove candidate: %w", err))
		}
		if hadPrevious {
			if err := os.Rename(backup, current); err != nil {
				recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback restore files: %w", err))
			} else if previousStateError == nil {
				if _, err := config.EnsureRuntime(ctx, previousState.Runtime); err != nil {
					recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback restart previous runtime: %w", err))
				}
			}
		}
		if controlChanged {
			if err := os.Remove(controlDestination); err != nil && !os.IsNotExist(err) {
				recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback controller: %w", err))
			}
			if controlHadPrevious {
				if err := os.Rename(controlBackup, controlDestination); err != nil {
					recoveryErrors = append(recoveryErrors, fmt.Errorf("rollback restore controller: %w", err))
				}
			}
		}
		if len(recoveryErrors) > 0 {
			resultErr = errors.Join(append([]error{resultErr}, recoveryErrors...)...)
		}
	}()
	if config.ControlBinarySource != "" {
		controlDestination = config.ControlBinaryDestination
		if controlDestination == "" {
			controlDestination = filepath.Join(config.InstallRoot, "bin", controlBinaryName(config.Platform.OS))
		}
		controlBackup, controlHadPrevious, controlChanged, err = persistControlBinary(config.ControlBinarySource, controlDestination)
		if err != nil {
			return fmt.Errorf("persist control binary: %w", err)
		}
	}
	runtimeStarted = true
	endpoint, err := config.EnsureRuntime(ctx, runtimeSpec)
	if err != nil {
		return fmt.Errorf("start runtime: %w", err)
	}
	previousProvider, installedProvider, err := mergeOpenCode(config.OpenCodeConfigPath, endpoint.BaseURL)
	if err != nil {
		return err
	}
	providerBeforeUpdate = previousProvider
	providerChanged = true
	originalProvider := previousProvider
	if previousStateError == nil {
		originalProvider = previousState.PreviousProvider
	}
	state := installState{
		SchemaVersion:     1,
		Release:           manifest.Release,
		ArtifactID:        releaseArtifactID,
		Runtime:           runtimeSpec,
		PreviousProvider:  originalProvider,
		InstalledProvider: installedProvider,
		RuntimeBaseURL:    endpoint.BaseURL,
		ControlBinary:     controlDestination,
		ModelSHA256:       modelSHA256,
		ModelSizeBytes:    modelSizeBytes,
		RuntimeSHA256:     runtimeSHA256,
		RuntimeSizeBytes:  runtimeSizeBytes,
		ServerSHA256:      serverSHA256,
		ServerSizeBytes:   serverSizeBytes,
	}
	if registerOllama {
		if config.RegisterOllama == nil {
			return errors.New("Ollama integration is unavailable")
		}
		modelfile, err := RenderOllamaModelfile(runtimeSpec.ModelPath)
		if err != nil {
			return err
		}
		profile := OllamaProfile{Name: productModelID, Modelfile: string(modelfile)}
		if err := config.RegisterOllama(ctx, profile); err != nil {
			return err
		}
		registeredProfile = profile
		ollamaRegistered = true
		state.Ollama = &profile
	}
	if err := writeJSON(statePath, state); err != nil {
		return err
	}
	committed = true
	_ = os.RemoveAll(backup)
	if controlBackup != "" {
		_ = os.Remove(controlBackup)
	}
	return nil
}

func uninstall(ctx context.Context, config CLIConfig) error {
	statePath := filepath.Join(config.InstallRoot, "install-state.json")
	state, err := readInstallState(statePath, config.InstallRoot)
	if err != nil {
		return err
	}
	if err := validateManagedControlBinary(state, config); err != nil {
		return err
	}
	if err := config.StopRuntime(ctx, state.Runtime); err != nil {
		return err
	}
	if config.CleanupRuntime != nil {
		if err := config.CleanupRuntime(ctx, state.Runtime); err != nil {
			return err
		}
	}
	// Ollama tags are mutable user-owned state. Automatic deletion could remove a
	// tag replaced after installation, so uninstall deliberately preserves it.
	if err := revertOpenCode(config.OpenCodeConfigPath, state); err != nil {
		return err
	}
	if err := os.RemoveAll(filepath.Join(config.InstallRoot, "current")); err != nil {
		return err
	}
	if err := os.Remove(statePath); err != nil {
		return err
	}
	if state.ControlBinary != "" && runtime.GOOS != "windows" {
		if err := os.Remove(state.ControlBinary); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func validateManagedControlBinary(state installState, config CLIConfig) error {
	if state.ControlBinary == "" {
		return nil
	}
	expected := config.ControlBinaryDestination
	if expected == "" {
		expected = filepath.Join(config.InstallRoot, "bin", controlBinaryName(config.Platform.OS))
	}
	if !managedExactPath(state.ControlBinary, expected, filepath.Dir(expected)) {
		return errors.New("install state control binary path is outside the managed destination")
	}
	return nil
}

func fetch(ctx context.Context, client *http.Client, rawURL string) ([]byte, error) {
	const maximumMetadataBytes = 8 * 1024 * 1024
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download %s: HTTP %d", rawURL, response.StatusCode)
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, maximumMetadataBytes+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > maximumMetadataBytes {
		return nil, errors.New("release metadata exceeds size limit")
	}
	return payload, nil
}

func downloadArtifact(
	ctx context.Context,
	client *http.Client,
	artifact Artifact,
	publicKey ed25519.PublicKey,
	destination string,
) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, artifact.URL, nil)
	if err != nil {
		return err
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download artifact: HTTP %d", response.StatusCode)
	}
	file, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	hash := sha256.New()
	written, copyErr := copyWithInactivityTimeout(
		io.MultiWriter(file, hash), response.Body, int64(artifact.SizeBytes)+1, artifactInactivityTimeout,
	)
	closeErr := file.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if written != int64(artifact.SizeBytes) {
		return errors.New("artifact size mismatch")
	}
	digest := hash.Sum(nil)
	if hex.EncodeToString(digest) != strings.ToLower(artifact.SHA256) {
		return errors.New("artifact SHA-256 mismatch")
	}
	signature, err := base64.StdEncoding.DecodeString(artifact.Signature)
	if err != nil || !ed25519.Verify(publicKey, digest, signature) {
		return errors.New("artifact signature verification failed")
	}
	return nil
}

func copyWithInactivityTimeout(destination io.Writer, source io.ReadCloser, maximumBytes int64, inactivity time.Duration) (int64, error) {
	if maximumBytes < 1 || inactivity <= 0 {
		return 0, errors.New("download limits are invalid")
	}
	activity := make(chan struct{}, 1)
	stop := make(chan struct{})
	watchdogDone := make(chan struct{})
	timedOut := make(chan struct{})
	go func() {
		defer close(watchdogDone)
		timer := time.NewTimer(inactivity)
		defer timer.Stop()
		for {
			select {
			case <-activity:
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(inactivity)
			case <-timer.C:
				close(timedOut)
				_ = source.Close()
				return
			case <-stop:
				return
			}
		}
	}()
	defer func() {
		close(stop)
		<-watchdogDone
	}()

	reader := io.LimitReader(source, maximumBytes)
	buffer := make([]byte, 1024*1024)
	var written int64
	for {
		count, readErr := reader.Read(buffer)
		if count > 0 {
			outputCount, writeErr := destination.Write(buffer[:count])
			written += int64(outputCount)
			if writeErr != nil {
				return written, writeErr
			}
			if outputCount != count {
				return written, io.ErrShortWrite
			}
			select {
			case activity <- struct{}{}:
			default:
			}
		}
		if readErr != nil {
			select {
			case <-timedOut:
				return written, errors.New("artifact download became inactive")
			default:
			}
			if errors.Is(readErr, io.EOF) {
				return written, nil
			}
			return written, readErr
		}
	}
}

func extractTarGzip(archivePath, destination string, maximumExpandedBytes uint64) error {
	if maximumExpandedBytes == 0 {
		return errors.New("runtime expanded-size limit is missing")
	}
	archive, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer archive.Close()
	gzipReader, err := gzip.NewReader(archive)
	if err != nil {
		return err
	}
	defer gzipReader.Close()
	reader := tar.NewReader(gzipReader)
	var expandedBytes uint64
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if header.Size < 0 || uint64(header.Size) > maximumExpandedBytes-expandedBytes {
			return errors.New("runtime archive exceeds its signed expanded-size limit")
		}
		expandedBytes += uint64(header.Size)
		clean := filepath.Clean(header.Name)
		if clean == "." || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return errors.New("artifact contains an unsafe path")
		}
		target := filepath.Join(destination, clean)
		if header.Typeflag != tar.TypeReg && header.Typeflag != tar.TypeRegA && header.Typeflag != tar.TypeDir {
			return errors.New("artifact contains an unsupported entry")
		}
		if header.Typeflag == tar.TypeDir {
			if err := os.MkdirAll(target, 0o700); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return err
		}
		mode := os.FileMode(0o600)
		if clean == filepath.Join("bin", "incubus-runtime") || clean == filepath.Join("bin", "incubus-runtime.exe") || clean == filepath.Join("bin", "llama-server") {
			mode = 0o700
		}
		file, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
		if err != nil {
			return err
		}
		_, copyErr := io.CopyN(file, reader, header.Size)
		closeErr := file.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
}

func activate(staging, current string) (string, bool, error) {
	backup := current + ".rollback"
	_ = os.RemoveAll(backup)
	hadCurrent := false
	if _, err := os.Stat(current); err == nil {
		hadCurrent = true
		if err := os.Rename(current, backup); err != nil {
			return "", false, err
		}
	} else if !os.IsNotExist(err) {
		return "", false, err
	}
	if err := os.Rename(staging, current); err != nil {
		if hadCurrent {
			_ = os.Rename(backup, current)
		}
		return "", false, err
	}
	return backup, hadCurrent, nil
}

func mergeOpenCode(path, baseURL string) (json.RawMessage, json.RawMessage, error) {
	document, err := readJSONObject(path)
	if err != nil {
		return nil, nil, err
	}
	providers, _ := document["provider"].(map[string]any)
	if providers == nil {
		providers = map[string]any{}
	}
	var previous json.RawMessage
	if value, exists := providers[productProviderID]; exists {
		previous, _ = json.Marshal(value)
	}
	providerJSON, err := RenderOpenCodeProviderJSON(OpenCodeProviderOptions{
		ProviderID: productProviderID, DisplayName: "Metaflora Incubus v1", BaseURL: baseURL,
		ModelID: productModelID, ModelName: "Metaflora Incubus v1",
	})
	if err != nil {
		return nil, nil, err
	}
	var rendered map[string]any
	if err := json.Unmarshal(providerJSON, &rendered); err != nil {
		return nil, nil, err
	}
	installed := rendered["provider"].(map[string]any)[productProviderID]
	providers[productProviderID] = installed
	document["provider"] = providers
	installedRaw, _ := json.Marshal(installed)
	return previous, installedRaw, writeJSON(path, document)
}

func revertOpenCode(path string, state installState) error {
	document, err := readJSONObject(path)
	if err != nil {
		return err
	}
	providers, ok := document["provider"].(map[string]any)
	if !ok {
		return errors.New("OpenCode provider ownership cannot be verified")
	}
	current, exists := providers[productProviderID]
	currentRaw, _ := json.Marshal(current)
	if !exists || !jsonEqual(currentRaw, state.InstalledProvider) {
		return errors.New("OpenCode provider changed after install; refusing to overwrite it")
	}
	if len(state.PreviousProvider) == 0 {
		delete(providers, productProviderID)
	} else {
		var previous any
		if err := json.Unmarshal(state.PreviousProvider, &previous); err != nil {
			return err
		}
		providers[productProviderID] = previous
	}
	document["provider"] = providers
	return writeJSON(path, document)
}

func readJSONObject(path string) (map[string]any, error) {
	payload, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return map[string]any{}, nil
	}
	if err != nil {
		return nil, err
	}
	var document map[string]any
	if err := json.Unmarshal(normalizeJSONC(payload), &document); err != nil {
		return nil, fmt.Errorf("invalid JSON config: %w", err)
	}
	return document, nil
}

func normalizeJSONC(payload []byte) []byte {
	withoutComments := append([]byte(nil), payload...)
	inString := false
	escaped := false
	for index := 0; index < len(withoutComments); index++ {
		current := withoutComments[index]
		if inString {
			if escaped {
				escaped = false
			} else if current == '\\' {
				escaped = true
			} else if current == '"' {
				inString = false
			}
			continue
		}
		if current == '"' {
			inString = true
			continue
		}
		if current != '/' || index+1 >= len(withoutComments) {
			continue
		}
		switch withoutComments[index+1] {
		case '/':
			withoutComments[index], withoutComments[index+1] = ' ', ' '
			index += 2
			for index < len(withoutComments) && withoutComments[index] != '\n' && withoutComments[index] != '\r' {
				withoutComments[index] = ' '
				index++
			}
			index--
		case '*':
			withoutComments[index], withoutComments[index+1] = ' ', ' '
			index += 2
			for index+1 < len(withoutComments) {
				if withoutComments[index] == '*' && withoutComments[index+1] == '/' {
					withoutComments[index], withoutComments[index+1] = ' ', ' '
					index++
					break
				}
				if withoutComments[index] != '\n' && withoutComments[index] != '\r' {
					withoutComments[index] = ' '
				}
				index++
			}
		}
	}
	result := make([]byte, 0, len(withoutComments))
	inString = false
	escaped = false
	for index := 0; index < len(withoutComments); index++ {
		current := withoutComments[index]
		if inString {
			result = append(result, current)
			if escaped {
				escaped = false
			} else if current == '\\' {
				escaped = true
			} else if current == '"' {
				inString = false
			}
			continue
		}
		if current == '"' {
			inString = true
			result = append(result, current)
			continue
		}
		if current == ',' {
			lookahead := index + 1
			for lookahead < len(withoutComments) && (withoutComments[lookahead] == ' ' || withoutComments[lookahead] == '\t' || withoutComments[lookahead] == '\n' || withoutComments[lookahead] == '\r') {
				lookahead++
			}
			if lookahead < len(withoutComments) && (withoutComments[lookahead] == '}' || withoutComments[lookahead] == ']') {
				continue
			}
		}
		result = append(result, current)
	}
	return result
}

func readInstallState(path, installRoot string) (installState, error) {
	var state installState
	expectedStatePath := filepath.Join(filepath.Clean(installRoot), "install-state.json")
	if !filepath.IsAbs(installRoot) || filepath.Clean(path) != expectedStatePath {
		return state, errors.New("install state path is outside the managed install root")
	}
	payload, err := os.ReadFile(path)
	if err != nil {
		return state, err
	}
	if err := json.Unmarshal(payload, &state); err != nil {
		return state, err
	}
	digests := []string{state.ModelSHA256, state.RuntimeSHA256, state.ServerSHA256}
	validDigests := true
	for _, encoded := range digests {
		digest, digestErr := hex.DecodeString(encoded)
		if digestErr != nil || len(digest) != sha256.Size {
			validDigests = false
		}
	}
	if state.SchemaVersion != 1 || state.Release == "" || state.ArtifactID == "" || state.Runtime.Host != "127.0.0.1" || !validDigests || state.ModelSizeBytes == 0 || state.RuntimeSizeBytes == 0 || state.ServerSizeBytes == 0 {
		return state, errors.New("invalid install state")
	}
	expectedModelPath := filepath.Join(filepath.Clean(installRoot), "current", "models", "incubus-v1.gguf")
	if !managedExactPath(state.Runtime.ModelPath, expectedModelPath, installRoot) {
		return state, errors.New("install state model path is outside the managed install root")
	}
	return state, nil
}

func managedExactPath(path, expected, root string) bool {
	if !filepath.IsAbs(path) || filepath.Clean(path) != filepath.Clean(expected) {
		return false
	}
	cleanRoot := filepath.Clean(root)
	relative, err := filepath.Rel(cleanRoot, filepath.Clean(path))
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return false
	}
	cursor := cleanRoot
	components := append([]string{"."}, strings.Split(relative, string(filepath.Separator))...)
	for _, component := range components {
		if component != "." {
			cursor = filepath.Join(cursor, component)
		}
		info, lstatErr := os.Lstat(cursor)
		if lstatErr == nil && info.Mode()&os.ModeSymlink != 0 {
			return false
		}
		if lstatErr != nil && !os.IsNotExist(lstatErr) {
			return false
		}
	}
	return true
}

func writeJSON(path string, value any) error {
	payload, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	payload = append(payload, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".incubus-")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
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
	return os.Rename(temporaryName, path)
}

func jsonEqual(left, right []byte) bool {
	var leftValue any
	var rightValue any
	return json.Unmarshal(left, &leftValue) == nil && json.Unmarshal(right, &rightValue) == nil && fmt.Sprint(leftValue) == fmt.Sprint(rightValue)
}

func containsArg(args []string, value string) bool {
	for _, argument := range args {
		if argument == value {
			return true
		}
	}
	return false
}

func artifactDigest(payload []byte) string {
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}
