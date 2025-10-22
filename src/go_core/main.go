package main

import (
	"context"
	"fmt"
	"log"
	"os"

	cfg "github.com/maastricht-university/edmo-pipeline/config"
	"github.com/maastricht-university/edmo-pipeline/orchestrator"
)

func main() {
	fmt.Println("EDMO Pipeline starting…")
	conf, err := cfg.Load()
	if err != nil {
		log.Fatal(err)
	}
	if len(os.Args) < 2 {
		log.Fatal("usage: pipeline <path/to/audio.wav>")
	}

	p := orchestrator.NewPipeline(conf)
	if err := p.Run(context.Background(), os.Args[1]); err != nil {
		log.Fatal(err)
	}
}
