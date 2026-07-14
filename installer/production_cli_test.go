package installer

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	state, err := readInstallState(filepath.Join(config.InstallRoot, "install-state.json"))
	if err != nil {
		t.Fatal(err)
	}
	if state.ControlBinary != destination {
		t.Fatalf("ControlBinary = %q, want %q", state.ControlBinary, destination)
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
