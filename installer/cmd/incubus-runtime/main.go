package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"syscall"
)

var modelIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,127}$`)

type runtimeConfiguration struct {
	modelPath string
	host      string
	port      int
	modelID   string
}

func (configuration runtimeConfiguration) serverArguments() []string {
	return []string{
		"--model", configuration.modelPath,
		"--host", configuration.host,
		"--port", strconv.Itoa(configuration.port),
		"--alias", configuration.modelID,
		"--jinja",
	}
}

func parseRuntimeArguments(arguments []string) (runtimeConfiguration, error) {
	flags := flag.NewFlagSet("incubus-runtime", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	var configuration runtimeConfiguration
	flags.StringVar(&configuration.modelPath, "model", "", "path to the installed GGUF")
	flags.StringVar(&configuration.host, "host", "", "runtime listen host")
	flags.IntVar(&configuration.port, "port", 0, "runtime listen port")
	flags.StringVar(&configuration.modelID, "model-id", "", "public model identifier")
	if err := flags.Parse(arguments); err != nil {
		return runtimeConfiguration{}, err
	}
	if flags.NArg() != 0 {
		return runtimeConfiguration{}, errors.New("unexpected positional runtime arguments")
	}
	if configuration.host != "127.0.0.1" {
		return runtimeConfiguration{}, errors.New("runtime must bind to IPv4 loopback")
	}
	if configuration.port < 1 || configuration.port > 65535 {
		return runtimeConfiguration{}, errors.New("runtime port is invalid")
	}
	if !modelIDPattern.MatchString(configuration.modelID) {
		return runtimeConfiguration{}, errors.New("model id is invalid")
	}
	model, err := os.Lstat(configuration.modelPath)
	if err != nil {
		return runtimeConfiguration{}, fmt.Errorf("inspect model: %w", err)
	}
	if !model.Mode().IsRegular() || model.Mode()&os.ModeSymlink != 0 {
		return runtimeConfiguration{}, errors.New("model must be a regular file")
	}
	configuration.modelPath, err = filepath.Abs(configuration.modelPath)
	if err != nil {
		return runtimeConfiguration{}, fmt.Errorf("resolve model path: %w", err)
	}
	return configuration, nil
}

func findLlamaServer(runtimeExecutable string) (string, error) {
	executable, err := filepath.Abs(runtimeExecutable)
	if err != nil {
		return "", err
	}
	server := filepath.Join(filepath.Dir(executable), "llama-server")
	info, err := os.Lstat(server)
	if err != nil {
		return "", fmt.Errorf("inspect bundled runtime server: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm()&0o111 == 0 {
		return "", errors.New("bundled runtime server must be a regular executable")
	}
	return server, nil
}

func run(arguments []string) error {
	configuration, err := parseRuntimeArguments(arguments)
	if err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	server, err := findLlamaServer(executable)
	if err != nil {
		return err
	}
	serverArguments := append([]string{server}, configuration.serverArguments()...)
	return syscall.Exec(server, serverArguments, os.Environ())
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
