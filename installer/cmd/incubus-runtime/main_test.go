package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestParseRuntimeArgumentsBuildsLoopbackLlamaServerCommand(t *testing.T) {
	model := filepath.Join(t.TempDir(), "incubus-v1.gguf")
	if err := os.WriteFile(model, []byte("gguf"), 0o600); err != nil {
		t.Fatal(err)
	}

	configuration, err := parseRuntimeArguments([]string{
		"--model", model,
		"--host", "127.0.0.1",
		"--port", "18991",
		"--model-id", "metaflora-incubus-v1",
	})
	if err != nil {
		t.Fatalf("parseRuntimeArguments() error = %v", err)
	}
	want := []string{
		"--model", model,
		"--host", "127.0.0.1",
		"--port", "18991",
		"--alias", "metaflora-incubus-v1",
		"--jinja",
	}
	if got := configuration.serverArguments(); !reflect.DeepEqual(got, want) {
		t.Fatalf("serverArguments() = %#v, want %#v", got, want)
	}
}

func TestParseRuntimeArgumentsRejectsUnsafeOrIncompleteInput(t *testing.T) {
	model := filepath.Join(t.TempDir(), "incubus-v1.gguf")
	if err := os.WriteFile(model, []byte("gguf"), 0o600); err != nil {
		t.Fatal(err)
	}
	valid := []string{"--model", model, "--host", "127.0.0.1", "--port", "18991", "--model-id", "metaflora-incubus-v1"}
	tests := []struct {
		name string
		args []string
	}{
		{name: "non-loopback host", args: replaceArgument(valid, "--host", "0.0.0.0")},
		{name: "invalid port", args: replaceArgument(valid, "--port", "70000")},
		{name: "invalid model id", args: replaceArgument(valid, "--model-id", "bad/model")},
		{name: "missing model", args: replaceArgument(valid, "--model", filepath.Join(t.TempDir(), "missing.gguf"))},
		{name: "unknown option", args: append(append([]string(nil), valid...), "--public")},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := parseRuntimeArguments(test.args); err == nil {
				t.Fatal("parseRuntimeArguments() accepted unsafe input")
			}
		})
	}
}

func TestFindLlamaServerRequiresSiblingRegularExecutable(t *testing.T) {
	directory := t.TempDir()
	runtimePath := filepath.Join(directory, "incubus-runtime")
	serverPath := filepath.Join(directory, "llama-server")
	if err := os.WriteFile(runtimePath, []byte("runtime"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(serverPath, []byte("server"), 0o700); err != nil {
		t.Fatal(err)
	}

	got, err := findLlamaServer(runtimePath)
	if err != nil {
		t.Fatalf("findLlamaServer() error = %v", err)
	}
	if got != serverPath {
		t.Fatalf("findLlamaServer() = %q, want %q", got, serverPath)
	}

	if err := os.Chmod(serverPath, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := findLlamaServer(runtimePath); err == nil || !strings.Contains(err.Error(), "executable") {
		t.Fatalf("findLlamaServer() error = %v", err)
	}
}

func replaceArgument(arguments []string, name, value string) []string {
	result := append([]string(nil), arguments...)
	for index := range result {
		if result[index] == name && index+1 < len(result) {
			result[index+1] = value
			return result
		}
	}
	return result
}
