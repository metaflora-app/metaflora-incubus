package installer

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

const pinnedRuntimeServerRevision = "bf2c86ddc0685f580595954056c2e77ebabfab4f"

func TestBuildRuntimeBundleProducesCompleteWorkingArtifact(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("runtime bundle is supported on macOS and Linux")
	}
	server := buildFakeInferenceServer(t)
	serverPayload, err := os.ReadFile(server)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(serverPayload)
	output := filepath.Join(t.TempDir(), "runtime.tar.gz")
	command := exec.Command(
		"sh", "build_runtime_bundle.sh", runtime.GOOS, runtime.GOARCH,
		server, hex.EncodeToString(digest[:]), filepath.Dir(server), output,
	)
	command.Env = append(os.Environ(), "PATH="+fakeGitPath(t, pinnedRuntimeServerRevision)+string(os.PathListSeparator)+os.Getenv("PATH"))
	if combined, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build runtime bundle: %v: %s", err, combined)
	}

	found := inspectRuntimeArchive(t, output)
	for _, name := range []string{"bin/incubus-runtime", "bin/llama-server", "metadata/runtime-revision"} {
		if !found[name] {
			t.Fatalf("bundle is missing %s", name)
		}
	}
	smokeRuntimeArchive(t, output)
}

func TestBuildRuntimeBundleRejectsUnpinnedOrMismatchedServer(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("runtime bundle is supported on macOS and Linux")
	}
	server := buildFakeInferenceServer(t)
	serverPayload, err := os.ReadFile(server)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(serverPayload)
	output := filepath.Join(t.TempDir(), "runtime.tar.gz")

	for _, test := range []struct {
		name     string
		revision string
		digest   string
	}{
		{name: "wrong revision", revision: "0123456789abcdef0123456789abcdef01234567", digest: hex.EncodeToString(digest[:])},
		{name: "wrong digest", revision: pinnedRuntimeServerRevision, digest: fmt.Sprintf("%064d", 0)},
	} {
		t.Run(test.name, func(t *testing.T) {
			command := exec.Command("sh", "build_runtime_bundle.sh", runtime.GOOS, runtime.GOARCH, server, test.digest, filepath.Dir(server), output)
			command.Env = append(os.Environ(), "PATH="+fakeGitPath(t, test.revision)+string(os.PathListSeparator)+os.Getenv("PATH"))
			if err := command.Run(); err == nil {
				t.Fatal("bundle builder accepted an unverified inference server")
			}
		})
	}
}

func TestBuildRuntimeBundleRejectsUnbundledInferenceLibraries(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("runtime bundle is supported on macOS and Linux")
	}
	server := buildFakeInferenceServer(t)
	payload, err := os.ReadFile(server)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(payload)
	tools := t.TempDir()
	inspector := "readelf"
	output := "Dynamic section: Shared library: [libllama.so]"
	if runtime.GOOS == "darwin" {
		inspector = "otool"
		output = "libllama.dylib"
	}
	inspectorScript := filepath.Join(tools, inspector)
	if err := os.WriteFile(inspectorScript, []byte("#!/bin/sh\nprintf '%s\\n' '"+output+"'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	gitScript := filepath.Join(tools, "git")
	if err := os.WriteFile(gitScript, []byte("#!/bin/sh\nprintf '%s\\n' '"+pinnedRuntimeServerRevision+"'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	archive := filepath.Join(t.TempDir(), "runtime.tar.gz")
	command := exec.Command("sh", "build_runtime_bundle.sh", runtime.GOOS, runtime.GOARCH, server, hex.EncodeToString(digest[:]), filepath.Dir(server), archive)
	command.Env = append(os.Environ(), "PATH="+tools+string(os.PathListSeparator)+os.Getenv("PATH"))
	if combined, err := command.CombinedOutput(); err == nil {
		t.Fatalf("bundle builder accepted dynamic inference libraries: %s", combined)
	}
}

func fakeGitPath(t *testing.T, revision string) string {
	t.Helper()
	directory := t.TempDir()
	script := filepath.Join(directory, "git")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nprintf '%s\\n' '"+revision+"'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	return directory
}

func buildFakeInferenceServer(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	source := filepath.Join(directory, "server.go")
	program := `package main
import ("encoding/json"; "flag"; "fmt"; "net/http"; "os")
func main() {
  for _, arg := range os.Args[1:] { if arg == "--version" { fmt.Println("test-server"); return } }
  model := flag.String("model", "", ""); host := flag.String("host", "", "")
  port := flag.Int("port", 0, ""); alias := flag.String("alias", "", ""); _ = flag.Bool("jinja", false, "")
  flag.Parse(); if *model == "" || *host != "127.0.0.1" || *alias == "" { os.Exit(2) }
  http.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": *alias}}}) })
  if err := http.ListenAndServe(fmt.Sprintf("%s:%d", *host, *port), nil); err != nil { os.Exit(1) }
}`
	if err := os.WriteFile(source, []byte(program), 0o600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(directory, "llama-server")
	command := exec.Command("go", "build", "-trimpath", "-o", binary, source)
	if combined, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build fake server: %v: %s", err, combined)
	}
	return binary
}

func inspectRuntimeArchive(t *testing.T, path string) map[string]bool {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gzipReader.Close()
	reader := tar.NewReader(gzipReader)
	found := map[string]bool{}
	for {
		header, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		found[header.Name] = true
		if header.Name == "bin/incubus-runtime" || header.Name == "bin/llama-server" {
			if header.FileInfo().Mode().Perm()&0o111 == 0 {
				t.Fatalf("%s is not executable", header.Name)
			}
		}
	}
	return found
}

func smokeRuntimeArchive(t *testing.T, archive string) {
	t.Helper()
	root := t.TempDir()
	if err := extractTarGzip(archive, root, 128*1024*1024); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(root, "model.gguf")
	if err := os.WriteFile(model, []byte("test model"), 0o600); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	runtimeCommand := exec.Command(
		filepath.Join(root, "bin", "incubus-runtime"),
		"--model", model, "--host", "127.0.0.1", "--port", fmt.Sprint(port), "--model-id", productModelID,
	)
	if err := runtimeCommand.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = runtimeCommand.Process.Kill()
		_ = runtimeCommand.Wait()
	}()
	client := &http.Client{Timeout: 200 * time.Millisecond}
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/models", port)
	deadline := time.Now().Add(5 * time.Second)
	for {
		response, requestErr := client.Get(url)
		if requestErr == nil {
			_ = response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("archived runtime did not become healthy: %v", requestErr)
		}
		time.Sleep(25 * time.Millisecond)
	}
}
