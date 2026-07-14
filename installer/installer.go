package installer

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strings"
)

const diskOverheadBytes = uint64(2 * 1024 * 1024 * 1024)
const maximumArtifactBytes = uint64(5 * 1024 * 1024 * 1024)
const splitInstallReserveBytes = uint64(256 * 1024 * 1024)

const (
	ArtifactRoleRuntime = "runtime"
	ArtifactRoleModel   = "model"
)

type Manifest struct {
	SchemaVersion int        `json:"schema_version"`
	Release       string     `json:"release"`
	ModelID       string     `json:"model_id"`
	Artifacts     []Artifact `json:"artifacts"`
}

type Artifact struct {
	ID                string `json:"id"`
	OS                string `json:"os"`
	Arch              string `json:"arch"`
	URL               string `json:"url"`
	SHA256            string `json:"sha256"`
	SizeBytes         uint64 `json:"size_bytes"`
	MinimumRAMBytes   uint64 `json:"minimum_ram_bytes"`
	Signature         string `json:"signature"`
	Format            string `json:"format"`
	Role              string `json:"role"`
	Revision          string `json:"revision"`
	UnpackedSizeBytes uint64 `json:"unpacked_size_bytes,omitempty"`
}

type ReleaseArtifacts struct {
	Runtime       Artifact
	Model         Artifact
	PeakDiskBytes uint64
}

type Platform struct {
	OS   string
	Arch string
}

type Resources struct {
	RAMBytes      uint64
	FreeDiskBytes uint64
}

type OpenCodeProviderOptions struct {
	ProviderID  string
	DisplayName string
	BaseURL     string
	ModelID     string
	ModelName   string
}

func ParseAndVerifyManifest(payload, signature []byte, publicKey ed25519.PublicKey) (Manifest, error) {
	if len(publicKey) != ed25519.PublicKeySize || len(signature) != ed25519.SignatureSize {
		return Manifest{}, errors.New("invalid Ed25519 key or signature size")
	}
	if !ed25519.Verify(publicKey, payload, signature) {
		return Manifest{}, errors.New("manifest signature verification failed")
	}
	var manifest Manifest
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return Manifest{}, fmt.Errorf("decode manifest: %w", err)
	}
	if manifest.SchemaVersion != 2 || manifest.Release == "" || manifest.ModelID == "" || len(manifest.Artifacts) < 2 {
		return Manifest{}, errors.New("invalid manifest metadata")
	}
	if manifest.ModelID != "metaflora-incubus-v1" {
		return Manifest{}, errors.New("manifest model id does not match this product")
	}
	seen := make(map[string]struct{}, len(manifest.Artifacts))
	modelCount := 0
	runtimeCount := 0
	for _, artifact := range manifest.Artifacts {
		parsed, err := url.Parse(artifact.URL)
		if err != nil || parsed.Scheme != "https" || parsed.Hostname() == "" || parsed.User != nil {
			return Manifest{}, fmt.Errorf("artifact %q must use a credential-free HTTPS URL", artifact.ID)
		}
		revision, revisionErr := hex.DecodeString(artifact.Revision)
		if artifact.ID == "" || artifact.SizeBytes == 0 || artifact.MinimumRAMBytes == 0 || revisionErr != nil || len(revision) != 20 || len(artifact.Revision) != 40 {
			return Manifest{}, fmt.Errorf("artifact %q has incomplete metadata", artifact.ID)
		}
		if _, duplicate := seen[artifact.ID]; duplicate {
			return Manifest{}, fmt.Errorf("duplicate artifact id %q", artifact.ID)
		}
		seen[artifact.ID] = struct{}{}
		switch artifact.Role {
		case ArtifactRoleModel:
			modelCount++
			if artifact.Format != "gguf" || artifact.UnpackedSizeBytes != 0 {
				return Manifest{}, errors.New("model artifact must be a direct GGUF download")
			}
		case ArtifactRoleRuntime:
			runtimeCount++
			if artifact.Format != "tar.gz" || artifact.UnpackedSizeBytes == 0 {
				return Manifest{}, errors.New("runtime artifact must declare its unpacked size")
			}
		default:
			return Manifest{}, fmt.Errorf("artifact %q has invalid role", artifact.ID)
		}
		digest, err := hex.DecodeString(artifact.SHA256)
		if err != nil || len(digest) != sha256.Size {
			return Manifest{}, fmt.Errorf("artifact %q has invalid SHA-256", artifact.ID)
		}
	}
	if modelCount != 1 || runtimeCount == 0 {
		return Manifest{}, errors.New("manifest must contain one model and at least one runtime")
	}
	return manifest, nil
}

func SelectReleaseArtifacts(manifest Manifest, platform Platform, resources Resources) (ReleaseArtifacts, error) {
	normalized, err := NormalizePlatform(platform.OS, platform.Arch)
	if err != nil {
		return ReleaseArtifacts{}, err
	}
	var model Artifact
	var runtimeArtifact Artifact
	for _, artifact := range manifest.Artifacts {
		switch artifact.Role {
		case ArtifactRoleModel:
			model = artifact
		case ArtifactRoleRuntime:
			candidate, normalizeErr := NormalizePlatform(artifact.OS, artifact.Arch)
			if normalizeErr == nil && candidate == normalized {
				runtimeArtifact = artifact
			}
		}
	}
	if model.ID == "" || runtimeArtifact.ID == "" {
		return ReleaseArtifacts{}, errors.New("no compatible split release artifacts")
	}
	minimumRAM := model.MinimumRAMBytes
	if runtimeArtifact.MinimumRAMBytes > minimumRAM {
		minimumRAM = runtimeArtifact.MinimumRAMBytes
	}
	if resources.RAMBytes < minimumRAM {
		return ReleaseArtifacts{}, errors.New("insufficient RAM")
	}
	peak, overflow := checkedSum(model.SizeBytes, runtimeArtifact.SizeBytes, runtimeArtifact.UnpackedSizeBytes, splitInstallReserveBytes)
	if overflow || peak > maximumArtifactBytes {
		return ReleaseArtifacts{}, errors.New("split release exceeds the 5 GiB peak-disk limit")
	}
	if resources.FreeDiskBytes < peak {
		return ReleaseArtifacts{}, errors.New("insufficient free disk for split release")
	}
	return ReleaseArtifacts{Runtime: runtimeArtifact, Model: model, PeakDiskBytes: peak}, nil
}

func checkedSum(values ...uint64) (uint64, bool) {
	var total uint64
	for _, value := range values {
		if value > ^uint64(0)-total {
			return 0, true
		}
		total += value
	}
	return total, false
}

func VerifySHA256(reader io.Reader, expected string) error {
	want, err := hex.DecodeString(expected)
	if err != nil || len(want) != sha256.Size {
		return errors.New("invalid expected SHA-256")
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, reader); err != nil {
		return fmt.Errorf("read artifact: %w", err)
	}
	if !equalBytes(hash.Sum(nil), want) {
		return errors.New("artifact SHA-256 mismatch")
	}
	return nil
}

func equalBytes(left, right []byte) bool {
	if len(left) != len(right) {
		return false
	}
	var mismatch byte
	for index := range left {
		mismatch |= left[index] ^ right[index]
	}
	return mismatch == 0
}

func SelectArtifact(manifest Manifest, platform Platform, resources Resources) (Artifact, error) {
	normalized, err := NormalizePlatform(platform.OS, platform.Arch)
	if err != nil {
		return Artifact{}, err
	}
	for _, artifact := range manifest.Artifacts {
		candidate, err := NormalizePlatform(artifact.OS, artifact.Arch)
		if err != nil || candidate != normalized {
			continue
		}
		if artifact.SizeBytes > maximumArtifactBytes {
			return Artifact{}, errors.New("release artifact exceeds the 5 GiB product limit")
		}
		if resources.RAMBytes < artifact.MinimumRAMBytes {
			return Artifact{}, errors.New("insufficient RAM")
		}
		if artifact.SizeBytes > ^uint64(0)-diskOverheadBytes || resources.FreeDiskBytes < artifact.SizeBytes+diskOverheadBytes {
			return Artifact{}, errors.New("insufficient free disk")
		}
		return artifact, nil
	}
	return Artifact{}, errors.New("no compatible artifact")
}

func NormalizePlatform(osName, arch string) (Platform, error) {
	osAliases := map[string]string{"darwin": "darwin", "macos": "darwin", "linux": "linux"}
	archAliases := map[string]string{"arm64": "arm64", "aarch64": "arm64", "amd64": "amd64", "x86_64": "amd64", "x64": "amd64"}
	osValue, osOK := osAliases[strings.ToLower(strings.TrimSpace(osName))]
	archValue, archOK := archAliases[strings.ToLower(strings.TrimSpace(arch))]
	if !osOK || !archOK {
		return Platform{}, fmt.Errorf("unsupported platform %s/%s", osName, arch)
	}
	return Platform{OS: osValue, Arch: archValue}, nil
}

func RenderOpenCodeProviderJSON(options OpenCodeProviderOptions) ([]byte, error) {
	if options.ProviderID == "" || options.DisplayName == "" || options.ModelID == "" || options.ModelName == "" {
		return nil, errors.New("provider and model identifiers are required")
	}
	baseURL, err := url.Parse(options.BaseURL)
	if err != nil || baseURL.Scheme != "http" || baseURL.Hostname() != "127.0.0.1" {
		return nil, errors.New("base URL must use loopback HTTP")
	}
	document := map[string]any{"provider": map[string]any{
		options.ProviderID: map[string]any{
			"npm":     "@ai-sdk/openai-compatible",
			"name":    options.DisplayName,
			"options": map[string]any{"baseURL": options.BaseURL},
			"models":  map[string]any{options.ModelID: map[string]any{"name": options.ModelName}},
		},
	}}
	return json.MarshalIndent(document, "", "  ")
}

func RenderOllamaModelfile(modelPath string) ([]byte, error) {
	if strings.ContainsAny(modelPath, "\r\n") || !filepath.IsAbs(modelPath) {
		return nil, errors.New("GGUF path must be absolute and contain no newlines")
	}
	quoted := strings.ReplaceAll(modelPath, `"`, `\"`)
	return []byte(fmt.Sprintf("FROM \"%s\"\nPARAMETER num_ctx 32768\n", quoted)), nil
}
