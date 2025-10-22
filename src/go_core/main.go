package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	cfg "github.com/maastricht-university/edmo-pipeline/config"
	"github.com/maastricht-university/edmo-pipeline/orchestrator"
)

func main() {
	fmt.Println("EDMO Pipeline starting…")

	audio := flag.String("audio", "", "path to audio file (wav/mp3/m4a)")
	flag.Parse()

	in := *audio
	if in == "" && flag.NArg() > 0 {
		in = flag.Arg(0)
	}
	if in == "" {
		log.Fatal("usage: pipeline [-audio path] <path/to/audio.(wav|mp3|m4a)>")
	}

	conf, err := cfg.Load()
	if err != nil {
		log.Fatal(err)
	}

	p := orchestrator.NewPipeline(conf)
	if err := p.Run(context.Background(), in); err != nil {
		log.Fatal(err)
	}
}
