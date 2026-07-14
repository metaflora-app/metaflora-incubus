package installer

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
)

const (
	giB = uint64(1024 * 1024 * 1024)
	miB = uint64(1024 * 1024)
)

func signedManifest(t *testing.T, manifestJSON string) ([]byte, []byte, ed25519.PublicKey) {
	t.Helper()

	seed := sha256.Sum256([]byte("metaflora-incubus-installer-test-key"))
	privateKey := ed25519.NewKeyFromSeed(seed[:])
	publicKey := privateKey.Public().(ed25519.PublicKey)
	payload := []byte(manifestJSON)
	signature := ed25519.Sign(privateKey, payload)

	return payload, signature, publicKey
}

func releaseManifestJSON(artifacts string) string {
	return `{
  "schema_version": 2,
  "release": "v1.0.0",
  "model_id": "metaflora-incubus-v1",
  "artifacts": [` + artifacts + `]
}`
}

func artifactJSON(id, osName, arch, url, digest string, size, minimumRAM uint64) string {
	return `{
  "id": ` + quote(id) + `,
  "os": ` + quote(osName) + `,
  "arch": ` + quote(arch) + `,
  "url": ` + quote(url) + `,
  "sha256": ` + quote(digest) + `,
  "size_bytes": ` + uintString(size) + `,
  "minimum_ram_bytes": ` + uintString(minimumRAM) + `
  ,"format": "tar.gz",
  "role": "runtime",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "unpacked_size_bytes": 1048576
}`
}

func modelArtifactJSON(digest string, size, minimumRAM uint64) string {
	return `{
  "id": "incubus-v1-q5",
  "os": "any",
  "arch": "any",
  "url": "https://huggingface.co/metaflora/incubus/resolve/0123456789abcdef0123456789abcdef01234567/incubus-v1.gguf",
  "sha256": ` + quote(digest) + `,
  "size_bytes": ` + uintString(size) + `,
  "minimum_ram_bytes": ` + uintString(minimumRAM) + `,
  "format": "gguf",
  "role": "model",
  "revision": "0123456789abcdef0123456789abcdef01234567"
}`
}

func quote(value string) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}

func uintString(value uint64) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}

func TestParseAndVerifyManifestAcceptsValidSignature(t *testing.T) {
	digest := strings.Repeat("a", 64)
	manifestJSON := releaseManifestJSON(artifactJSON(
		"macos-arm64-q5",
		"darwin",
		"arm64",
		"https://downloads.metaflora.ai/incubus/v1/macos-arm64.tar.zst",
		digest,
		5*giB,
		12*giB,
	) + "," + modelArtifactJSON(strings.Repeat("e", 64), 3*giB, 12*giB))
	payload, signature, publicKey := signedManifest(t, manifestJSON)

	manifest, err := ParseAndVerifyManifest(payload, signature, publicKey)
	if err != nil {
		t.Fatalf("ParseAndVerifyManifest() error = %v", err)
	}
	if manifest.Release != "v1.0.0" {
		t.Fatalf("Release = %q, want v1.0.0", manifest.Release)
	}
	if manifest.ModelID != "metaflora-incubus-v1" {
		t.Fatalf("ModelID = %q, want metaflora-incubus-v1", manifest.ModelID)
	}
	if len(manifest.Artifacts) != 2 {
		t.Fatalf("len(Artifacts) = %d, want 2", len(manifest.Artifacts))
	}
}

func TestParseAndVerifyManifestRejectsTampering(t *testing.T) {
	digest := strings.Repeat("b", 64)
	payload, signature, publicKey := signedManifest(t, releaseManifestJSON(artifactJSON(
		"linux-amd64-q5",
		"linux",
		"amd64",
		"https://downloads.metaflora.ai/incubus/v1/linux-amd64.tar.zst",
		digest,
		5*giB,
		12*giB,
	)))
	tampered := bytes.Replace(payload, []byte("v1.0.0"), []byte("v1.0.1"), 1)

	if _, err := ParseAndVerifyManifest(tampered, signature, publicKey); err == nil {
		t.Fatal("ParseAndVerifyManifest() accepted a manifest modified after signing")
	}
}

func TestParseAndVerifyManifestRejectsNonHTTPSArtifactURL(t *testing.T) {
	digest := strings.Repeat("c", 64)
	payload, signature, publicKey := signedManifest(t, releaseManifestJSON(artifactJSON(
		"linux-amd64-q5",
		"linux",
		"amd64",
		"http://downloads.metaflora.ai/incubus/v1/linux-amd64.tar.zst",
		digest,
		5*giB,
		12*giB,
	)))

	if _, err := ParseAndVerifyManifest(payload, signature, publicKey); err == nil {
		t.Fatal("ParseAndVerifyManifest() accepted a non-HTTPS artifact URL")
	}
}

func TestParseAndVerifyManifestRejectsDifferentProductID(t *testing.T) {
	digest := strings.Repeat("d", 64)
	document := strings.Replace(releaseManifestJSON(artifactJSON(
		"linux-amd64-q5", "linux", "amd64",
		"https://downloads.metaflora.ai/incubus/v1/linux-amd64.tar.zst",
		digest, 5*giB, 12*giB,
	)), "metaflora-incubus-v1", "different-product", 1)
	payload, signature, publicKey := signedManifest(t, document)
	if _, err := ParseAndVerifyManifest(payload, signature, publicKey); err == nil {
		t.Fatal("ParseAndVerifyManifest() accepted a different product id")
	}
}

func TestVerifySHA256(t *testing.T) {
	payload := []byte("immutable incubus artifact")
	digest := sha256.Sum256(payload)

	if err := VerifySHA256(bytes.NewReader(payload), hex.EncodeToString(digest[:])); err != nil {
		t.Fatalf("VerifySHA256() error = %v", err)
	}
}

func TestVerifySHA256RejectsCorruptArtifact(t *testing.T) {
	expected := sha256.Sum256([]byte("expected artifact"))

	err := VerifySHA256(bytes.NewReader([]byte("corrupt artifact")), hex.EncodeToString(expected[:]))
	if err == nil {
		t.Fatal("VerifySHA256() accepted corrupt artifact bytes")
	}
}

func TestSelectArtifactMatchesPlatformAndResources(t *testing.T) {
	manifest := Manifest{Artifacts: []Artifact{
		{
			ID:              "macos-amd64-q5",
			OS:              "darwin",
			Arch:            "amd64",
			SizeBytes:       5 * giB,
			MinimumRAMBytes: 12 * giB,
		},
		{
			ID:              "macos-arm64-q5",
			OS:              "darwin",
			Arch:            "arm64",
			SizeBytes:       5 * giB,
			MinimumRAMBytes: 12 * giB,
		},
	}}

	artifact, err := SelectArtifact(manifest, Platform{OS: "darwin", Arch: "arm64"}, Resources{
		RAMBytes:      16 * giB,
		FreeDiskBytes: 14 * giB,
	})
	if err != nil {
		t.Fatalf("SelectArtifact() error = %v", err)
	}
	if artifact.ID != "macos-arm64-q5" {
		t.Fatalf("Artifact.ID = %q, want macos-arm64-q5", artifact.ID)
	}
}

func TestSelectArtifactRefusesInsufficientResources(t *testing.T) {
	manifest := Manifest{Artifacts: []Artifact{{
		ID:              "macos-arm64-q5",
		OS:              "darwin",
		Arch:            "arm64",
		SizeBytes:       5 * giB,
		MinimumRAMBytes: 12 * giB,
	}}}
	platform := Platform{OS: "darwin", Arch: "arm64"}

	tests := []struct {
		name      string
		resources Resources
	}{
		{
			name: "RAM below minimum",
			resources: Resources{
				RAMBytes:      8 * giB,
				FreeDiskBytes: 14 * giB,
			},
		},
		{
			name: "disk cannot hold download and installation overhead",
			resources: Resources{
				RAMBytes:      16 * giB,
				FreeDiskBytes: 5*giB + 512*miB,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := SelectArtifact(manifest, platform, test.resources); err == nil {
				t.Fatal("SelectArtifact() accepted insufficient host resources")
			}
		})
	}
}

func TestSelectArtifactRejectsArtifactLargerThanFiveGiB(t *testing.T) {
	manifest := Manifest{Artifacts: []Artifact{{
		ID: "oversized", OS: "darwin", Arch: "arm64",
		SizeBytes: 5*giB + 1, MinimumRAMBytes: 1,
	}}}
	_, err := SelectArtifact(manifest, Platform{OS: "darwin", Arch: "arm64"}, Resources{
		RAMBytes: 16 * giB, FreeDiskBytes: 32 * giB,
	})
	if err == nil || !strings.Contains(err.Error(), "5 GiB") {
		t.Fatalf("SelectArtifact() error = %v", err)
	}
}

func TestSelectReleaseArtifactsEnforcesCombinedPeakDiskBudget(t *testing.T) {
	revision := "0123456789abcdef0123456789abcdef01234567"
	manifest := Manifest{Artifacts: []Artifact{
		{ID: "runtime", Role: ArtifactRoleRuntime, OS: "darwin", Arch: "arm64", Format: "tar.gz", Revision: revision, SizeBytes: 64 * miB, UnpackedSizeBytes: 128 * miB, MinimumRAMBytes: 1},
		{ID: "model", Role: ArtifactRoleModel, OS: "any", Arch: "any", Format: "gguf", Revision: revision, SizeBytes: 4 * giB, MinimumRAMBytes: 1},
	}}
	selected, err := SelectReleaseArtifacts(manifest, Platform{OS: "darwin", Arch: "arm64"}, Resources{RAMBytes: 16 * giB, FreeDiskBytes: 5 * giB})
	if err != nil {
		t.Fatalf("SelectReleaseArtifacts() error = %v", err)
	}
	if selected.Runtime.ID != "runtime" || selected.Model.ID != "model" || selected.PeakDiskBytes > 5*giB {
		t.Fatalf("unexpected selection: %#v", selected)
	}

	manifest.Artifacts[1].SizeBytes = 5 * giB
	if _, err := SelectReleaseArtifacts(manifest, Platform{OS: "darwin", Arch: "arm64"}, Resources{RAMBytes: 16 * giB, FreeDiskBytes: 16 * giB}); err == nil {
		t.Fatal("SelectReleaseArtifacts() accepted a release whose peak exceeds 5 GiB")
	}
}

func TestNormalizePlatform(t *testing.T) {
	tests := []struct {
		name     string
		osName   string
		arch     string
		expected Platform
	}{
		{name: "Go macOS ARM", osName: "darwin", arch: "arm64", expected: Platform{OS: "darwin", Arch: "arm64"}},
		{name: "human macOS spelling", osName: "macos", arch: "aarch64", expected: Platform{OS: "darwin", Arch: "arm64"}},
		{name: "Linux x86 alias", osName: "linux", arch: "x86_64", expected: Platform{OS: "linux", Arch: "amd64"}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual, err := NormalizePlatform(test.osName, test.arch)
			if err != nil {
				t.Fatalf("NormalizePlatform() error = %v", err)
			}
			if actual != test.expected {
				t.Fatalf("NormalizePlatform() = %#v, want %#v", actual, test.expected)
			}
		})
	}
}

func TestNormalizePlatformRejectsUnsupportedPlatform(t *testing.T) {
	if _, err := NormalizePlatform("plan9", "mips"); err == nil {
		t.Fatal("NormalizePlatform() accepted an unsupported platform")
	}
}

func TestRenderOpenCodeProviderJSON(t *testing.T) {
	configuration, err := RenderOpenCodeProviderJSON(OpenCodeProviderOptions{
		ProviderID:  "metaflora-incubus",
		DisplayName: "Metaflora Incubus v1",
		BaseURL:     "http://127.0.0.1:8080/v1",
		ModelID:     "metaflora-incubus-v1",
		ModelName:   "Metaflora Incubus v1",
	})
	if err != nil {
		t.Fatalf("RenderOpenCodeProviderJSON() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(configuration, &decoded); err != nil {
		t.Fatalf("generated config is invalid JSON: %v", err)
	}
	provider := decoded["provider"].(map[string]any)["metaflora-incubus"].(map[string]any)
	if provider["npm"] != "@ai-sdk/openai-compatible" {
		t.Fatalf("provider npm = %q", provider["npm"])
	}
	if provider["name"] != "Metaflora Incubus v1" {
		t.Fatalf("provider name = %q", provider["name"])
	}
	options := provider["options"].(map[string]any)
	if options["baseURL"] != "http://127.0.0.1:8080/v1" {
		t.Fatalf("baseURL = %q", options["baseURL"])
	}
	models := provider["models"].(map[string]any)
	model := models["metaflora-incubus-v1"].(map[string]any)
	if model["name"] != "Metaflora Incubus v1" {
		t.Fatalf("model name = %q", model["name"])
	}
	if _, exists := options["apiKey"]; exists {
		t.Fatal("local OpenCode provider must not contain an API key")
	}
}

func TestRenderOllamaModelfileUsesInstalledGGUF(t *testing.T) {
	modelfile, err := RenderOllamaModelfile("/Users/example/.local/share/metaflora-incubus/model/incubus-v1.gguf")
	if err != nil {
		t.Fatalf("RenderOllamaModelfile() error = %v", err)
	}
	rendered := string(modelfile)
	if !strings.Contains(rendered, `FROM "/Users/example/.local/share/metaflora-incubus/model/incubus-v1.gguf"`) {
		t.Fatalf("Modelfile does not point at installed GGUF: %s", rendered)
	}
	if strings.Contains(strings.ToLower(rendered), "api_key") {
		t.Fatal("local Ollama profile must not contain an API key")
	}
}

func TestRenderOllamaModelfileRejectsDirectiveInjection(t *testing.T) {
	if _, err := RenderOllamaModelfile("/tmp/incubus.gguf\nSYSTEM compromised"); err == nil {
		t.Fatal("RenderOllamaModelfile() accepted a newline in the model path")
	}
}
