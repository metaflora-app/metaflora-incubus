package installer

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"sync"
	"testing"
)

type fakeRelease struct {
	client        *http.Client
	manifestURL   string
	signatureURL  string
	publicKey     ed25519.PublicKey
	requests      map[string]int
	requestsMutex *sync.Mutex
	close         func()
}

type releaseFaults struct {
	badManifestSignature bool
	badArtifactSignature bool
	badArtifactHash      bool
	badArtifactSize      bool
	missingRuntime       bool
}

func newFakeRelease(t *testing.T, faults releaseFaults) fakeRelease {
	return newFakeReleaseVersion(t, "v1.0.0", faults)
}

func newFakeReleaseVersion(t *testing.T, version string, faults releaseFaults) fakeRelease {
	t.Helper()

	seed := sha256.Sum256([]byte("incubus-v1-test-release-key"))
	privateKey := ed25519.NewKeyFromSeed(seed[:])
	publicKey := privateKey.Public().(ed25519.PublicKey)
	runtimeArtifact := testBundleVariant(t, !faults.missingRuntime)
	modelArtifact := []byte("test model weights")
	runtimeDigest := sha256.Sum256(runtimeArtifact)
	modelDigest := sha256.Sum256(modelArtifact)
	runtimeHash := hex.EncodeToString(runtimeDigest[:])
	modelHash := hex.EncodeToString(modelDigest[:])
	runtimeSize := len(runtimeArtifact)
	modelSize := len(modelArtifact)
	runtimeSignature := ed25519.Sign(privateKey, runtimeDigest[:])
	modelSignature := ed25519.Sign(privateKey, modelDigest[:])
	if faults.badArtifactHash {
		modelHash = strings.Repeat("0", sha256.Size*2)
	}
	if faults.badArtifactSize {
		modelSize++
	}
	if faults.badArtifactSignature {
		modelSignature[0] ^= 0xff
	}

	requests := make(map[string]int)
	requestsMutex := &sync.Mutex{}
	var manifest []byte
	var manifestSignature []byte
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requestsMutex.Lock()
		requests[request.URL.Path]++
		requestsMutex.Unlock()
		switch request.URL.Path {
		case "/release/manifest.json":
			_, _ = writer.Write(manifest)
		case "/release/manifest.json.sig":
			_, _ = writer.Write(manifestSignature)
		case "/release/incubus-runtime.tar.gz":
			_, _ = writer.Write(runtimeArtifact)
		case "/release/incubus-v1.gguf":
			_, _ = writer.Write(modelArtifact)
		default:
			http.NotFound(writer, request)
		}
	}))

	manifestDocument := map[string]any{
		"schema_version": 2,
		"release":        version,
		"model_id":       "metaflora-incubus-v1",
		"artifacts": []map[string]any{{
			"id":                  "test-runtime",
			"os":                  "darwin",
			"arch":                "arm64",
			"url":                 server.URL + "/release/incubus-runtime.tar.gz",
			"sha256":              runtimeHash,
			"signature":           base64.StdEncoding.EncodeToString(runtimeSignature),
			"size_bytes":          runtimeSize,
			"minimum_ram_bytes":   1,
			"format":              "tar.gz",
			"role":                "runtime",
			"revision":            "0123456789abcdef0123456789abcdef01234567",
			"unpacked_size_bytes": 4096,
		}, {
			"id":                "test-model-q5",
			"os":                "any",
			"arch":              "any",
			"url":               server.URL + "/release/incubus-v1.gguf",
			"sha256":            modelHash,
			"signature":         base64.StdEncoding.EncodeToString(modelSignature),
			"size_bytes":        modelSize,
			"minimum_ram_bytes": 1,
			"format":            "gguf",
			"role":              "model",
			"revision":          "0123456789abcdef0123456789abcdef01234567",
		}},
	}
	var err error
	manifest, err = json.Marshal(manifestDocument)
	if err != nil {
		t.Fatalf("marshal release manifest: %v", err)
	}
	manifestSignature = ed25519.Sign(privateKey, manifest)
	if faults.badManifestSignature {
		manifestSignature[0] ^= 0xff
	}

	return fakeRelease{
		client:        server.Client(),
		manifestURL:   server.URL + "/release/manifest.json",
		signatureURL:  server.URL + "/release/manifest.json.sig",
		publicKey:     publicKey,
		requests:      requests,
		requestsMutex: requestsMutex,
		close:         server.Close,
	}
}

func (release *fakeRelease) requestCount(path string) int {
	release.requestsMutex.Lock()
	defer release.requestsMutex.Unlock()
	return release.requests[path]
}

func testBundle(t *testing.T) []byte {
	return testBundleVariant(t, true)
}

func testBundleVariant(t *testing.T, includeRuntime bool) []byte {
	t.Helper()

	var archive bytes.Buffer
	gzipWriter := gzip.NewWriter(&archive)
	tarWriter := tar.NewWriter(gzipWriter)
	files := map[string]string{
		"legal/THIRD_PARTY_NOTICES": "required legal notices",
	}
	if includeRuntime {
		files["bin/incubus-runtime"] = "test runtime"
		files["bin/llama-server"] = "test inference server"
	}
	for name, contents := range files {
		header := &tar.Header{Name: name, Mode: 0o600, Size: int64(len(contents))}
		if err := tarWriter.WriteHeader(header); err != nil {
			t.Fatalf("write tar header: %v", err)
		}
		if _, err := io.WriteString(tarWriter, contents); err != nil {
			t.Fatalf("write tar contents: %v", err)
		}
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatalf("close tar: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("close gzip: %v", err)
	}
	return archive.Bytes()
}

type runtimeHarness struct {
	server      *httptest.Server
	ensureCalls int
	stopCalls   int
	lastSpec    RuntimeSpec
}

func (runtime *runtimeHarness) ensure(_ context.Context, spec RuntimeSpec) (RuntimeEndpoint, error) {
	runtime.ensureCalls++
	runtime.lastSpec = spec
	if spec.Host != "127.0.0.1" {
		return RuntimeEndpoint{}, errors.New("runtime requested a non-loopback bind")
	}
	if runtime.server == nil {
		runtime.server = httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			if request.URL.Path != "/v1/models" {
				http.NotFound(writer, request)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"object":"list","data":[{"id":"metaflora-incubus-v1"}]}`)
		}))
	}
	return RuntimeEndpoint{BaseURL: runtime.server.URL + "/v1"}, nil
}

func (runtime *runtimeHarness) stop(context.Context, RuntimeSpec) error {
	runtime.stopCalls++
	if runtime.server != nil {
		runtime.server.Close()
		runtime.server = nil
	}
	return nil
}

type ollamaHarness struct {
	registerCalls   int
	unregisterCalls int
	profile         OllamaProfile
}

func (ollama *ollamaHarness) register(_ context.Context, profile OllamaProfile) error {
	ollama.registerCalls++
	ollama.profile = profile
	return nil
}

func (ollama *ollamaHarness) unregister(context.Context, OllamaProfile) error {
	ollama.unregisterCalls++
	return nil
}

func testCLIConfig(t *testing.T, release fakeRelease, runtime *runtimeHarness, ollama *ollamaHarness) CLIConfig {
	t.Helper()
	root := t.TempDir()
	return CLIConfig{
		ManifestURL:          release.manifestURL,
		ManifestSignatureURL: release.signatureURL,
		PinnedPublicKey:      release.publicKey,
		HTTPClient:           release.client,
		Platform:             Platform{OS: "darwin", Arch: "arm64"},
		ProbeResources: func(context.Context, string) (Resources, error) {
			return Resources{RAMBytes: 16 * giB, FreeDiskBytes: 16 * giB}, nil
		},
		InstallRoot:        filepath.Join(root, "incubus"),
		OpenCodeConfigPath: filepath.Join(root, "opencode", "opencode.json"),
		EnsureRuntime:      runtime.ensure,
		StopRuntime:        runtime.stop,
		RegisterOllama:     ollama.register,
		UnregisterOllama:   ollama.unregister,
	}
}

func runCLI(t *testing.T, config CLIConfig, args ...string) (int, string, string) {
	t.Helper()
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	exitCode := RunCLI(context.Background(), args, &stdout, &stderr, config)
	return exitCode, stdout.String(), stderr.String()
}

func TestHeadlessInstallFromSignedLocalRelease(t *testing.T) {
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
	foreignConfig := `{"provider":{"keep-me":{"name":"Foreign provider"}},"theme":"dark"}`
	if err := os.MkdirAll(filepath.Dir(config.OpenCodeConfigPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(config.OpenCodeConfigPath, []byte(foreignConfig), 0o600); err != nil {
		t.Fatal(err)
	}

	exitCode, stdout, stderr := runCLI(t, config, "install", "--non-interactive", "--register-ollama")
	if exitCode != 0 {
		t.Fatalf("install exit code = %d\nstdout: %s\nstderr: %s", exitCode, stdout, stderr)
	}
	if release.requestCount("/public-key") != 0 {
		t.Fatal("installer downloaded a public key instead of using its pinned key")
	}
	for _, relativePath := range []string{"bin/incubus-runtime", "models/incubus-v1.gguf", "config/runtime.json"} {
		if _, err := os.Stat(filepath.Join(config.InstallRoot, "current", relativePath)); err != nil {
			t.Errorf("activated file %q: %v", relativePath, err)
		}
	}
	if runtimeInfo, err := os.Stat(filepath.Join(config.InstallRoot, "current", "bin", "incubus-runtime")); err != nil {
		t.Fatal(err)
	} else if goruntime.GOOS != "windows" && runtimeInfo.Mode().Perm()&0o100 == 0 {
		t.Fatalf("runtime mode = %o, executable bit is missing", runtimeInfo.Mode().Perm())
	}
	if matches, err := filepath.Glob(filepath.Join(config.InstallRoot, ".staging-*")); err != nil || len(matches) != 0 {
		t.Fatalf("staging directories after activation = %v, err = %v", matches, err)
	}
	if runtime.ensureCalls != 1 || runtime.lastSpec.Host != "127.0.0.1" {
		t.Fatalf("runtime ensure calls/spec = %d/%+v", runtime.ensureCalls, runtime.lastSpec)
	}
	response, err := http.Get(runtime.server.URL + "/v1/models")
	if err != nil {
		t.Fatalf("OpenAI-compatible endpoint is unavailable after install: %v", err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/models status = %d", response.StatusCode)
	}

	configuration, err := os.ReadFile(config.OpenCodeConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(configuration, &decoded); err != nil {
		t.Fatalf("OpenCode config is invalid JSON: %v", err)
	}
	providers := decoded["provider"].(map[string]any)
	if _, ok := providers["keep-me"]; !ok {
		t.Fatal("OpenCode merge deleted an unrelated provider")
	}
	incubus := providers["metaflora-incubus"].(map[string]any)
	options := incubus["options"].(map[string]any)
	if _, ok := options["apiKey"]; ok {
		t.Fatal("local OpenCode config contains an API key")
	}
	baseURL, _ := options["baseURL"].(string)
	if !strings.HasPrefix(baseURL, "http://127.0.0.1:") {
		t.Fatalf("OpenCode baseURL = %q, want loopback", baseURL)
	}
	if ollama.registerCalls != 1 || ollama.profile.Name != "metaflora-incubus-v1" {
		t.Fatalf("Ollama registration = %d/%+v", ollama.registerCalls, ollama.profile)
	}
	assertNoBuildInputNames(t, stdout+stderr+string(configuration)+ollama.profile.Modelfile)
}

func TestInstallIsIdempotent(t *testing.T) {
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

	for attempt := 1; attempt <= 2; attempt++ {
		code, stdout, stderr := runCLI(t, config, "install", "--non-interactive")
		if code != 0 {
			t.Fatalf("install attempt %d failed: %s%s", attempt, stdout, stderr)
		}
	}
	if got := release.requestCount("/release/incubus-runtime.tar.gz"); got != 1 {
		t.Fatalf("artifact download count = %d, want 1", got)
	}
	if got := release.requestCount("/release/incubus-v1.gguf"); got != 1 {
		t.Fatalf("direct GGUF download count = %d, want 1", got)
	}
	if runtime.ensureCalls != 2 {
		t.Fatalf("runtime ensure count = %d, want 2 healthy idempotent checks", runtime.ensureCalls)
	}
}

func TestSameReleaseFastPathRefusesModifiedWeights(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	runtime := &runtimeHarness{}
	defer func() {
		if runtime.server != nil {
			runtime.server.Close()
		}
	}()
	config := testCLIConfig(t, release, runtime, &ollamaHarness{})
	if code, _, stderr := runCLI(t, config, "install", "--non-interactive"); code != 0 {
		t.Fatalf("install failed: %s", stderr)
	}
	state, err := readInstallState(filepath.Join(config.InstallRoot, "install-state.json"), config.InstallRoot)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(state.Runtime.ModelPath, []byte("tampered"), 0o600); err != nil {
		t.Fatal(err)
	}
	config.ProbeResources = func(context.Context, string) (Resources, error) {
		return Resources{RAMBytes: 16 * giB, FreeDiskBytes: 0}, nil
	}
	ensureCalls := runtime.ensureCalls
	code, _, stderr := runCLI(t, config, "install", "--non-interactive")
	if code == 0 || !strings.Contains(stderr, "integrity") {
		t.Fatalf("same-release install exit=%d stderr=%q", code, stderr)
	}
	if runtime.ensureCalls != ensureCalls {
		t.Fatal("runtime was started before same-release integrity validation")
	}
}

func TestRuntimeArchiveMustContainExecutableBeforeActivation(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{missingRuntime: true})
	defer release.close()
	runtime := &runtimeHarness{}
	config := testCLIConfig(t, release, runtime, &ollamaHarness{})
	code, _, stderr := runCLI(t, config, "install", "--non-interactive")
	if code == 0 || !strings.Contains(stderr, "required regular executable") {
		t.Fatalf("install exit=%d stderr=%q", code, stderr)
	}
	if runtime.ensureCalls != 0 {
		t.Fatal("invalid runtime archive touched active runtime")
	}
	if _, err := os.Stat(filepath.Join(config.InstallRoot, "current")); !os.IsNotExist(err) {
		t.Fatalf("invalid runtime archive activated files: %v", err)
	}
}

func TestInstallRunsInjectableResourcePreflightBeforeDownload(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	runtime := &runtimeHarness{}
	ollama := &ollamaHarness{}
	config := testCLIConfig(t, release, runtime, ollama)
	probeCalls := 0
	config.ProbeResources = func(_ context.Context, installRoot string) (Resources, error) {
		probeCalls++
		if installRoot != config.InstallRoot {
			t.Fatalf("probe install root = %q", installRoot)
		}
		return Resources{}, errors.New("not enough local resources")
	}

	code, _, _ := runCLI(t, config, "install", "--non-interactive")
	if code == 0 || probeCalls != 1 {
		t.Fatalf("exit/probe calls = %d/%d, want nonzero/1", code, probeCalls)
	}
	if got := release.requestCount("/release/incubus-runtime.tar.gz"); got != 0 {
		t.Fatalf("artifact downloaded %d times despite failed preflight", got)
	}
}

func TestInstallRejectsEveryReleaseIntegrityFailure(t *testing.T) {
	tests := []struct {
		name   string
		faults releaseFaults
	}{
		{name: "manifest signature", faults: releaseFaults{badManifestSignature: true}},
		{name: "artifact signature", faults: releaseFaults{badArtifactSignature: true}},
		{name: "artifact hash", faults: releaseFaults{badArtifactHash: true}},
		{name: "artifact size", faults: releaseFaults{badArtifactSize: true}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			release := newFakeRelease(t, test.faults)
			defer release.close()
			runtime := &runtimeHarness{}
			ollama := &ollamaHarness{}
			config := testCLIConfig(t, release, runtime, ollama)

			code, _, _ := runCLI(t, config, "install", "--non-interactive")
			if code == 0 {
				t.Fatal("installer accepted a release with failed integrity verification")
			}
			if _, err := os.Stat(filepath.Join(config.InstallRoot, "current")); !os.IsNotExist(err) {
				t.Fatalf("failed release was activated: %v", err)
			}
			if runtime.ensureCalls != 0 {
				t.Fatal("runtime was started for an untrusted release")
			}
		})
	}
}

func TestActivationFailureRollsBackPreviousRelease(t *testing.T) {
	firstRelease := newFakeReleaseVersion(t, "v1.0.0", releaseFaults{})
	defer firstRelease.close()
	runtime := &runtimeHarness{}
	defer func() {
		if runtime.server != nil {
			runtime.server.Close()
		}
	}()
	ollama := &ollamaHarness{}
	config := testCLIConfig(t, firstRelease, runtime, ollama)
	if code, stdout, stderr := runCLI(t, config, "install", "--non-interactive"); code != 0 {
		t.Fatalf("initial install failed: %s%s", stdout, stderr)
	}

	current := filepath.Join(config.InstallRoot, "current")
	marker := filepath.Join(current, "previous-release")
	if err := os.WriteFile(marker, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	secondRelease := newFakeReleaseVersion(t, "v1.0.1", releaseFaults{})
	defer secondRelease.close()
	config.ManifestURL = secondRelease.manifestURL
	config.ManifestSignatureURL = secondRelease.signatureURL
	config.PinnedPublicKey = secondRelease.publicKey
	config.HTTPClient = secondRelease.client
	activationCalls := 0
	config.BeforeActivate = func(stagingDir, targetDir string) error {
		activationCalls++
		if !strings.Contains(filepath.Base(stagingDir), ".staging-") || targetDir != current {
			t.Fatalf("unexpected activation paths %q -> %q", stagingDir, targetDir)
		}
		return errors.New("injected activation failure")
	}

	code, _, _ := runCLI(t, config, "install", "--non-interactive")
	if code == 0 {
		t.Fatal("install succeeded despite activation failure")
	}
	if activationCalls != 1 {
		t.Fatalf("activation hook calls = %d, want 1", activationCalls)
	}
	contents, err := os.ReadFile(marker)
	if err != nil || string(contents) != "keep" {
		t.Fatalf("previous release was not preserved: %q, %v", contents, err)
	}
	if matches, err := filepath.Glob(filepath.Join(config.InstallRoot, ".staging-*")); err != nil || len(matches) != 0 {
		t.Fatalf("failed staging directories = %v, err = %v", matches, err)
	}
}

func TestRuntimeFailureAfterActivationRollsBackPreviousRelease(t *testing.T) {
	firstRelease := newFakeReleaseVersion(t, "v1.0.0", releaseFaults{})
	defer firstRelease.close()
	runtimeHarness := &runtimeHarness{}
	defer func() {
		if runtimeHarness.server != nil {
			runtimeHarness.server.Close()
		}
	}()
	ollama := &ollamaHarness{}
	config := testCLIConfig(t, firstRelease, runtimeHarness, ollama)
	if code, stdout, stderr := runCLI(t, config, "install", "--non-interactive"); code != 0 {
		t.Fatalf("initial install failed: %s%s", stdout, stderr)
	}
	marker := filepath.Join(config.InstallRoot, "current", "previous-release")
	if err := os.WriteFile(marker, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}

	secondRelease := newFakeReleaseVersion(t, "v1.0.1", releaseFaults{})
	defer secondRelease.close()
	config.ManifestURL = secondRelease.manifestURL
	config.ManifestSignatureURL = secondRelease.signatureURL
	config.PinnedPublicKey = secondRelease.publicKey
	config.HTTPClient = secondRelease.client
	ensureCalls := 0
	config.EnsureRuntime = func(context.Context, RuntimeSpec) (RuntimeEndpoint, error) {
		ensureCalls++
		return RuntimeEndpoint{}, errors.New("injected runtime failure")
	}

	code, _, stderr := runCLI(t, config, "install", "--non-interactive")
	if code == 0 {
		t.Fatal("update succeeded despite runtime failure")
	}
	if contents, err := os.ReadFile(marker); err != nil || string(contents) != "keep" {
		t.Fatalf("previous release was not restored: %q, %v", contents, err)
	}
	if ensureCalls != 2 {
		t.Fatalf("runtime ensure calls = %d, want failed update plus rollback restart", ensureCalls)
	}
	if !strings.Contains(stderr, "rollback restart previous runtime") {
		t.Fatalf("rollback restart failure was hidden: %s", stderr)
	}
}

func TestFailedReplacementDownloadKeepsOldModelUntouched(t *testing.T) {
	initial := newFakeReleaseVersion(t, "v1.0.0", releaseFaults{})
	defer initial.close()
	runtime := &runtimeHarness{}
	defer func() {
		if runtime.server != nil {
			runtime.server.Close()
		}
	}()
	config := testCLIConfig(t, initial, runtime, &ollamaHarness{})
	if code, _, stderr := runCLI(t, config, "install", "--non-interactive"); code != 0 {
		t.Fatalf("initial install failed: %s", stderr)
	}
	state, err := readInstallState(filepath.Join(config.InstallRoot, "install-state.json"), config.InstallRoot)
	if err != nil {
		t.Fatal(err)
	}

	broken := newFakeReleaseVersion(t, "v1.1.0", releaseFaults{badArtifactHash: true})
	defer broken.close()
	config.ManifestURL = broken.manifestURL
	config.ManifestSignatureURL = broken.signatureURL
	config.PinnedPublicKey = broken.publicKey
	config.HTTPClient = broken.client
	if code, _, _ := runCLI(t, config, "update"); code == 0 {
		t.Fatal("update unexpectedly accepted corrupt replacement weights")
	}
	payload, err := os.ReadFile(state.Runtime.ModelPath)
	if err != nil || string(payload) != "test model weights" {
		t.Fatalf("old GGUF was changed after failed replacement: %q, %v", payload, err)
	}
	if _, err := os.Stat(filepath.Join(config.InstallRoot, "current", "bin", "incubus-runtime")); err != nil {
		t.Fatalf("previous runtime was not preserved: %v", err)
	}
}

func TestLowDiskUpdateRequiresExplicitUninstallInsteadOfDeletingOldGGUF(t *testing.T) {
	initial := newFakeReleaseVersion(t, "v1.0.0", releaseFaults{})
	defer initial.close()
	runtime := &runtimeHarness{}
	defer func() {
		if runtime.server != nil {
			runtime.server.Close()
		}
	}()
	config := testCLIConfig(t, initial, runtime, &ollamaHarness{})
	if code, _, stderr := runCLI(t, config, "install", "--non-interactive"); code != 0 {
		t.Fatalf("initial install failed: %s", stderr)
	}

	next := newFakeReleaseVersion(t, "v1.1.0", releaseFaults{})
	defer next.close()
	config.ManifestURL = next.manifestURL
	config.ManifestSignatureURL = next.signatureURL
	config.PinnedPublicKey = next.publicKey
	config.HTTPClient = next.client
	// This is deliberately below the signed runtime+model+reserve peak. Avoid
	// recompressing a map-backed test archive here: tar entry order can change
	// its compressed size and make the boundary assertion flaky.
	freeBeforeRemoval := splitInstallReserveBytes
	config.ProbeResources = func(context.Context, string) (Resources, error) {
		return Resources{RAMBytes: 16 * giB, FreeDiskBytes: freeBeforeRemoval}, nil
	}
	code, _, stderr := runCLI(t, config, "update")
	if code == 0 || !strings.Contains(stderr, "uninstall") {
		t.Fatalf("low-disk update exit=%d stderr=%q", code, stderr)
	}
	state, err := readInstallState(filepath.Join(config.InstallRoot, "install-state.json"), config.InstallRoot)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(state.Runtime.ModelPath); err != nil {
		t.Fatalf("low-disk refusal removed active GGUF: %v", err)
	}
	if got := next.requestCount("/release/incubus-runtime.tar.gz"); got != 0 {
		t.Fatalf("low-disk refusal downloaded runtime %d times", got)
	}
}

func TestUninstallRemovesOnlyInstallerOwnedState(t *testing.T) {
	release := newFakeRelease(t, releaseFaults{})
	defer release.close()
	runtime := &runtimeHarness{}
	ollama := &ollamaHarness{}
	config := testCLIConfig(t, release, runtime, ollama)
	if code, stdout, stderr := runCLI(t, config, "install", "--non-interactive", "--register-ollama"); code != 0 {
		t.Fatalf("install failed: %s%s", stdout, stderr)
	}
	unowned := filepath.Join(filepath.Dir(config.InstallRoot), "user-file.txt")
	if err := os.WriteFile(unowned, []byte("do not delete"), 0o600); err != nil {
		t.Fatal(err)
	}
	configuration, err := os.ReadFile(config.OpenCodeConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(configuration, &decoded); err != nil {
		t.Fatal(err)
	}
	decoded["provider"].(map[string]any)["keep-me"] = map[string]any{"name": "Foreign provider"}
	configuration, _ = json.Marshal(decoded)
	if err := os.WriteFile(config.OpenCodeConfigPath, configuration, 0o600); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := runCLI(t, config, "uninstall", "--non-interactive")
	if code != 0 {
		t.Fatalf("uninstall failed: %s%s", stdout, stderr)
	}
	if contents, err := os.ReadFile(unowned); err != nil || string(contents) != "do not delete" {
		t.Fatalf("unowned file was modified: %q, %v", contents, err)
	}
	configuration, err = os.ReadFile(config.OpenCodeConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(configuration), "metaflora-incubus") || !strings.Contains(string(configuration), "keep-me") {
		t.Fatalf("uninstall changed the wrong OpenCode provider: %s", configuration)
	}
	if runtime.stopCalls != 1 || ollama.unregisterCalls != 0 {
		t.Fatalf("runtime cleanup/Ollama preservation calls = %d/%d", runtime.stopCalls, ollama.unregisterCalls)
	}
}

func assertNoBuildInputNames(t *testing.T, value string) {
	t.Helper()
	lower := strings.ToLower(value)
	for _, forbidden := range []string{
		strings.Join([]string{"q", "wen"}, ""),
		strings.Join([]string{"deep", "seek"}, ""),
		strings.Join([]string{"ki", "mi"}, ""),
	} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("user-visible output or config leaked a build-input name: %q", forbidden)
		}
	}
}
