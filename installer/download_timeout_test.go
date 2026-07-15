package installer

import (
	"bytes"
	"io"
	"strings"
	"testing"
	"time"
)

func TestCopyWithInactivityTimeoutStopsStalledBody(t *testing.T) {
	reader, writer := io.Pipe()
	go func() {
		_, _ = writer.Write([]byte("partial"))
		// The read side must close this stalled writer when the inactivity
		// deadline expires.
		<-time.After(time.Second)
		_ = writer.Close()
	}()
	started := time.Now()
	var destination bytes.Buffer
	_, err := copyWithInactivityTimeout(&destination, reader, 1024, 30*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "inactive") {
		t.Fatalf("copyWithInactivityTimeout() error = %v", err)
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("stalled body took %s to abort", elapsed)
	}
	if destination.String() != "partial" {
		t.Fatalf("destination = %q", destination.String())
	}
}

func TestCopyWithInactivityTimeoutAcceptsActiveBody(t *testing.T) {
	source := io.NopCloser(strings.NewReader("complete artifact"))
	var destination bytes.Buffer
	written, err := copyWithInactivityTimeout(&destination, source, 1024, time.Second)
	if err != nil {
		t.Fatalf("copyWithInactivityTimeout() error = %v", err)
	}
	if written != int64(len("complete artifact")) || destination.String() != "complete artifact" {
		t.Fatalf("written=%d destination=%q", written, destination.String())
	}
}
