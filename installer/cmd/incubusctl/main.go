package main

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"fmt"
	"os"

	installer "github.com/metaflora-app/metaflora-incubus/installer"
)

// Set by the release build with -ldflags "-X main.pinnedPublicKeyBase64=...".
var pinnedPublicKeyBase64 string

func main() {
	key, err := base64.StdEncoding.DecodeString(pinnedPublicKeyBase64)
	if err != nil || len(key) != ed25519.PublicKeySize {
		fmt.Fprintln(os.Stderr, "incubusctl release trust key is missing or invalid; refusing to continue")
		os.Exit(2)
	}
	config, err := installer.NewProductionCLIConfig(installer.ProductionOptions{PinnedPublicKey: ed25519.PublicKey(key)})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	os.Exit(installer.RunCLI(context.Background(), os.Args[1:], os.Stdout, os.Stderr, config))
}
