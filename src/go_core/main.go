package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	cfg "github.com/maastricht-university/edmo-pipeline/config"
	"github.com/maastricht-university/edmo-pipeline/orchestrator"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("[edmo] ")
	start := time.Now()
	log.Println("EDMO Pipeline starting…")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: pipeline [-audio path] <audio.(wav|mp3|m4a)>\n")
		flag.PrintDefaults()
	}
	audio := flag.String("audio", "", "path to audio file (wav/mp3/m4a)")
	flag.Parse()

	in := *audio
	if in == "" && flag.NArg() > 0 {
		in = flag.Arg(0)
	}
	if abs, err := filepath.Abs(in); err == nil {
		in = abs
	}

	if in == "" {
		flag.Usage()
		os.Exit(2)
	}

	if st, err := os.Stat(in); err != nil || st.IsDir() {
		log.Fatalf("input not found or is a directory: %s", in)
	}

	switch ext := strings.ToLower(filepath.Ext(in)); ext {
	case ".wav", ".mp3", ".m4a":
	default:
		log.Printf("warning: unexpected extension %q; proceeding anyway", ext)
	}

	conf, err := cfg.Load()
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM, syscall.SIGQUIT)
	defer stop()

	p := orchestrator.NewPipeline(conf)
	if err := p.Run(ctx, in); err != nil {
		if ctx.Err() != nil {
			log.Printf("stopped: %v", ctx.Err())
		} else {
			log.Fatal(err)
		}
	}
	log.Printf("done in %s", time.Since(start).Round(time.Millisecond))
}
