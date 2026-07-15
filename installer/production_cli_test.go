package installer

import (
	"context"
	"crypto/ed25519"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"testing"
)

func TestInstallPersistsControlBinary(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	runtime := &runtimeHarness{}
	defer func() {
		if runtime.server != nil {
			runtime.server.Close()
		}
	}()
	ollama := &ollamaHarness{}
	config := testCLIConfig(t, release, runtime, ollama)
	source := filepath.Join(t.TempDir(), "incubusctl")
	if err := os.WriteFile(source, []byte("signed controller bytes"), 0o700); err != nil {
		t.Fatal(err)
	}
	config.ControlBinarySource = source

	exitCode, _, stderr := runCLI(t, config, "install", "--non-interactive")
	if exitCode != 0 {
		t.Fatalf("install failed: %s", stderr)
	}
	destination := filepath.Join(config.InstallRoot, "bin", controlBinaryName(config.Platform.OS))
	payload, err := os.ReadFile(destination)
	if err != nil {
		t.Fatalf("read persisted controller: %v", err)
	}
	if string(payload) != "signed controller bytes" {
		t.Fatalf("persisted controller = %q", payload)
	}
	state, err := readInstallState(filepath.Join(config.InstallRoot, "install-state.json"), config.InstallRoot)
	if err != nil {
		t.Fatal(err)
	}
	if state.ControlBinary != destination {
		t.Fatalf("ControlBinary = %q, want %q", state.ControlBinary, destination)
	}
}

func TestProductionHTTPClientDoesNotApplyMetadataTimeoutToModelBody(t *testing.T) {
	publicKey, _, keyErr := ed25519.GenerateKey(nil)
	if keyErr != nil {
		t.Fatal(keyErr)
	}
	executable := filepath.Join(t.TempDir(), "incubusctl")
	if err := os.WriteFile(executable, []byte("controller"), 0o700); err != nil {
		t.Fatal(err)
	}
	config, err := NewProductionCLIConfig(ProductionOptions{
		PinnedPublicKey: publicKey,
		Executable:      executable,
	})
	if err != nil {
		t.Fatal(err)
	}
	if config.HTTPClient.Timeout != 0 {
		t.Fatalf("HTTP client timeout = %s; multi-gigabyte response bodies must not share a metadata deadline", config.HTTPClient.Timeout)
	}
	transport, ok := config.HTTPClient.Transport.(*http.Transport)
	if !ok || transport.ResponseHeaderTimeout <= 0 {
		t.Fatalf("HTTP transport = %#v; want a bounded response-header timeout", config.HTTPClient.Transport)
	}
}

func TestFailedInstallRestoresPreviousControlBinary(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	config := testCLIConfig(t, release, &runtimeHarness{}, &ollamaHarness{})
	destination := filepath.Join(config.InstallRoot, "bin", controlBinaryName(config.Platform.OS))
	if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(destination, []byte("previous controller"), 0o700); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(t.TempDir(), "incubusctl")
	if err := os.WriteFile(source, []byte("new controller"), 0o700); err != nil {
		t.Fatal(err)
	}
	config.ControlBinarySource = source
	config.EnsureRuntime = func(context.Context, RuntimeSpec) (RuntimeEndpoint, error) {
		return RuntimeEndpoint{}, errors.New("synthetic service failure")
	}

	if exitCode, _, _ := runCLI(t, config, "install", "--non-interactive"); exitCode == 0 {
		t.Fatal("install succeeded despite runtime failure")
	}
	payload, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(payload) != "previous controller" {
		t.Fatalf("rollback left %q", payload)
	}
}

func TestOperationalCommandsUsePersistedState(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	runtime := &runtimeHarness{}
	defer func() {
		if runtime.server != nil {
			runtime.server.Close()
		}
	}()
	config := testCLIConfig(t, release, runtime, &ollamaHarness{})
	if exitCode, _, stderr := runCLI(t, config, "install", "--non-interactive"); exitCode != 0 {
		t.Fatalf("install failed: %s", stderr)
	}

	if exitCode, stdout, stderr := runCLI(t, config, "status"); exitCode != 0 || !strings.Contains(stdout, "running") {
		t.Fatalf("status = %d stdout=%q stderr=%q", exitCode, stdout, stderr)
	}
	if exitCode, _, stderr := runCLI(t, config, "stop"); exitCode != 0 {
		t.Fatalf("stop failed: %s", stderr)
	}
	if exitCode, _, stderr := runCLI(t, config, "start"); exitCode != 0 {
		t.Fatalf("start failed: %s", stderr)
	}
	if exitCode, _, stderr := runCLI(t, config, "restart"); exitCode != 0 {
		t.Fatalf("restart failed: %s", stderr)
	}
	if runtime.stopCalls != 2 {
		t.Fatalf("stop calls = %d, want 2", runtime.stopCalls)
	}
}

func TestUnimplementedCommandsFailExplicitly(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	config := testCLIConfig(t, release, &runtimeHarness{}, &ollamaHarness{})
	for _, command := range []string{"rollback", "profile", "bundle", "move"} {
		t.Run(command, func(t *testing.T) {
			exitCode, _, stderr := runCLI(t, config, command)
			if exitCode == 0 || !strings.Contains(strings.ToLower(stderr), "unsupported") {
				t.Fatalf("%s exit=%d stderr=%q", command, exitCode, stderr)
			}
		})
	}
}

func TestDoctorReportsMissingInstallHonestly(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	config := testCLIConfig(t, release, &runtimeHarness{}, &ollamaHarness{})
	exitCode, _, stderr := runCLI(t, config, "doctor")
	if exitCode == 0 || !strings.Contains(stderr, "not installed") {
		t.Fatalf("doctor exit=%d stderr=%q", exitCode, stderr)
	}
}

func TestInstallStateRejectsModelPathOutsideManagedRoot(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(filepath.Dir(root), "foreign.gguf")
	state := installState{
		SchemaVersion: 1, Release: "v1", ArtifactID: "runtime+model",
		Runtime:     RuntimeSpec{Host: "127.0.0.1", ModelPath: outside, ModelID: productModelID},
		ModelSHA256: strings.Repeat("a", 64), ModelSizeBytes: 1,
		RuntimeSHA256: strings.Repeat("b", 64), RuntimeSizeBytes: 1,
		ServerSHA256: strings.Repeat("c", 64), ServerSizeBytes: 1,
	}
	path := filepath.Join(root, "install-state.json")
	if err := writeJSON(path, state); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(outside); err == nil {
		t.Fatal("test outside path unexpectedly exists")
	}
	if _, err := readInstallState(path, root); err == nil || !strings.Contains(err.Error(), "outside") {
		t.Fatalf("readInstallState() error = %v", err)
	}
}

func TestInstallStateRejectsSymlinkedModelParent(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root, "current")); err != nil {
		if goruntime.GOOS == "windows" {
			t.Skipf("symlink unavailable: %v", err)
		}
		t.Fatal(err)
	}
	state := installState{
		SchemaVersion: 1, Release: "v1", ArtifactID: "runtime+model",
		Runtime:     RuntimeSpec{Host: "127.0.0.1", ModelPath: filepath.Join(root, "current", "models", "incubus-v1.gguf"), ModelID: productModelID},
		ModelSHA256: strings.Repeat("a", 64), ModelSizeBytes: 1,
		RuntimeSHA256: strings.Repeat("b", 64), RuntimeSizeBytes: 1,
		ServerSHA256: strings.Repeat("c", 64), ServerSizeBytes: 1,
	}
	path := filepath.Join(root, "install-state.json")
	if err := writeJSON(path, state); err != nil {
		t.Fatal(err)
	}
	if _, err := readInstallState(path, root); err == nil || !strings.Contains(err.Error(), "outside") {
		t.Fatalf("readInstallState() error = %v", err)
	}
}

func TestUninstallRejectsControlBinaryOutsideManagedDestination(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	config := testCLIConfig(t, release, &runtimeHarness{}, &ollamaHarness{})
	if exitCode, _, stderr := runCLI(t, config, "install", "--non-interactive"); exitCode != 0 {
		t.Fatalf("install failed: %s", stderr)
	}
	statePath := filepath.Join(config.InstallRoot, "install-state.json")
	state, err := readInstallState(statePath, config.InstallRoot)
	if err != nil {
		t.Fatal(err)
	}
	foreign := filepath.Join(t.TempDir(), "foreign-user-file")
	if err := os.WriteFile(foreign, []byte("keep me"), 0o600); err != nil {
		t.Fatal(err)
	}
	state.ControlBinary = foreign
	if err := writeJSON(statePath, state); err != nil {
		t.Fatal(err)
	}

	exitCode, _, stderr := runCLI(t, config, "uninstall")
	if exitCode == 0 || !strings.Contains(stderr, "control binary") {
		t.Fatalf("uninstall exit=%d stderr=%q", exitCode, stderr)
	}
	if payload, err := os.ReadFile(foreign); err != nil || string(payload) != "keep me" {
		t.Fatalf("foreign file changed: payload=%q err=%v", payload, err)
	}
}

func TestMissingLaunchdServiceIsIdempotent(t *testing.T) {
	for _, message := range []string{"Could not find service", "No such process", "service not found"} {
		if !isMissingLaunchdService(errors.New(message)) {
			t.Fatalf("missing-service error was not recognized: %q", message)
		}
	}
	if isMissingLaunchdService(errors.New("permission denied")) {
		t.Fatal("permission failure was mistaken for an absent service")
	}
}

func TestValidateStagedRuntimeRequiresBundledInferenceServer(t *testing.T) {
	staging := t.TempDir()
	bin := filepath.Join(staging, "bin")
	if err := os.MkdirAll(bin, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bin, "incubus-runtime"), []byte("wrapper"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := validateStagedRuntime(staging, "darwin"); err == nil || !strings.Contains(err.Error(), "server") {
		t.Fatalf("validateStagedRuntime() error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(bin, "llama-server"), []byte("server"), 0o700); err != nil {
		t.Fatal(err)
	}
	legal := filepath.Join(staging, "legal")
	if err := os.MkdirAll(legal, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"LICENSE", "THIRD_PARTY_NOTICES"} {
		if err := os.WriteFile(filepath.Join(legal, name), []byte("reviewed legal text"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := validateStagedRuntime(staging, "darwin"); err != nil {
		t.Fatalf("validateStagedRuntime() error = %v", err)
	}
}

func TestDoctorDetectsModifiedInstalledWeights(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	runtime := &runtimeHarness{}
	defer func() {
		if runtime.server != nil {
			runtime.server.Close()
		}
	}()
	config := testCLIConfig(t, release, runtime, &ollamaHarness{})
	if exitCode, _, stderr := runCLI(t, config, "install", "--non-interactive"); exitCode != 0 {
		t.Fatalf("install failed: %s", stderr)
	}
	state, err := readInstallState(filepath.Join(config.InstallRoot, "install-state.json"), config.InstallRoot)
	if err != nil {
		t.Fatal(err)
	}
	if state.ModelSHA256 == "" || state.ModelSizeBytes == 0 {
		t.Fatalf("model integrity was not persisted: %#v", state)
	}
	if err := os.WriteFile(state.Runtime.ModelPath, []byte("modified weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	exitCode, _, stderr := runCLI(t, config, "doctor")
	if exitCode == 0 || !strings.Contains(strings.ToLower(stderr), "integrity") {
		t.Fatalf("doctor exit=%d stderr=%q", exitCode, stderr)
	}
}

func TestDoctorDetectsModifiedBundledInferenceServer(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	runtimeHarness := &runtimeHarness{}
	defer func() {
		if runtimeHarness.server != nil {
			runtimeHarness.server.Close()
		}
	}()
	config := testCLIConfig(t, release, runtimeHarness, &ollamaHarness{})
	if exitCode, _, stderr := runCLI(t, config, "install", "--non-interactive"); exitCode != 0 {
		t.Fatalf("install failed: %s", stderr)
	}
	server := filepath.Join(config.InstallRoot, "current", "bin", "llama-server")
	if err := os.WriteFile(server, []byte("modified inference server"), 0o700); err != nil {
		t.Fatal(err)
	}
	exitCode, _, stderr := runCLI(t, config, "doctor")
	if exitCode == 0 || !strings.Contains(strings.ToLower(stderr), "integrity") {
		t.Fatalf("doctor exit=%d stderr=%q", exitCode, stderr)
	}
}

func TestValidateHostedArtifactRequiresDirectMetafloraIncubusDownload(t *testing.T) {
	tests := []struct {
		name    string
		rawURL  string
		wantErr bool
	}{
		{name: "immutable hosted release", rawURL: "https://huggingface.co/metaflora/incubus/resolve/0123456789abcdef0123456789abcdef01234567/incubus-v1.tar.gz"},
		{name: "mutable main revision", rawURL: "https://huggingface.co/metaflora/incubus/resolve/main/incubus-v1.tar.gz", wantErr: true},
		{name: "credential query", rawURL: "https://huggingface.co/metaflora/incubus/resolve/0123456789abcdef0123456789abcdef01234567/incubus-v1.tar.gz?token=secret", wantErr: true},
		{name: "wrong repository", rawURL: "https://huggingface.co/foreign/model/resolve/main/incubus-v1.tar.gz", wantErr: true},
		{name: "lookalike host", rawURL: "https://huggingface.co.evil.test/metaflora/incubus/resolve/main/file", wantErr: true},
		{name: "local build file", rawURL: "file:///tmp/incubus-v1.tar.gz", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateHostedArtifact(Artifact{URL: test.rawURL, SizeBytes: 5 * giB, Revision: "0123456789abcdef0123456789abcdef01234567"})
			if (err != nil) != test.wantErr {
				t.Fatalf("ValidateHostedArtifact() error = %v, wantErr=%v", err, test.wantErr)
			}
		})
	}
}

func TestReadJSONObjectAcceptsJSONCWithoutChangingForeignValues(t *testing.T) {
	path := filepath.Join(t.TempDir(), "opencode.jsonc")
	payload := `{
  // user comment
  "theme": "dark", /* keep the semantic value */
  "provider": {
    "foreign": {"name": "https://example.test//literal",},
  },
}`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	document, err := readJSONObject(path)
	if err != nil {
		t.Fatalf("readJSONObject() error = %v", err)
	}
	if document["theme"] != "dark" {
		t.Fatalf("theme = %#v", document["theme"])
	}
	provider := document["provider"].(map[string]any)["foreign"].(map[string]any)
	if provider["name"] != "https://example.test//literal" {
		t.Fatalf("string literal was damaged: %#v", provider)
	}
}

func TestServiceCommandRunnerDoesNotInvokeShell(t *testing.T) {
	runner := ExecServiceCommandRunner{}
	command := ServiceCommand{Name: "", Args: []string{"sh", "-c", "evil"}}
	if err := runner.Run(context.Background(), command); err == nil {
		t.Fatal("runner accepted an empty executable name")
	}
}

func TestProductionConfigPersistsControllerInUserLocalBin(t *testing.T) {
	key := make([]byte, 32)
	executable := filepath.Join(t.TempDir(), "incubusctl")
	config, err := NewProductionCLIConfig(ProductionOptions{
		PinnedPublicKey: key,
		Executable:      executable,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(
		config.ControlBinaryDestination,
		filepath.Join(".local", "bin", controlBinaryName(config.Platform.OS)),
	) {
		t.Fatalf("unexpected controller destination: %s", config.ControlBinaryDestination)
	}
}

func TestHealthCheckRejectsUnrelatedHTTP200Service(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"data":[{"id":"some-other-model"}]}`))
	}))
	defer server.Close()

	if err := checkRuntimeHealth(context.Background(), server.Client(), server.URL+"/v1"); err == nil {
		t.Fatal("unrelated HTTP 200 service passed the runtime health check")
	}
}
