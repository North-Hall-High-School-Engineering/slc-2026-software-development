//go:build windows
// +build windows

package audio

import (
	"fmt"
)

// type Loopback interface
type windowsLoopback struct{}

func newLoopback() Loopback {
	return &windowsLoopback{}
}

func (w *windowsLoopback) Start() error {
	return fmt.Errorf("not implemented")
}

func (w *windowsLoopback) Stop() error {
	return fmt.Errorf("not implemented")
}

func (w *windowsLoopback) Read() ([]float32, error) {
	return nil, fmt.Errorf("not implemented")
}
