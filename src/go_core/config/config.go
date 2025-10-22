package config

import (
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

type Service struct {
	URL string `yaml:"url"`
}
type Services struct {
	NLP           Service `yaml:"nlp"`
	ASR           Service `yaml:"asr"`
	Diarization   Service `yaml:"diarization"`
	Emotion       Service `yaml:"emotion"`
	Clustering    Service `yaml:"clustering"`
	Visualization Service `yaml:"visualization"`
}
type Audio struct {
	SampleRate int    `yaml:"sample_rate"`
	Channels   int    `yaml:"channels"`
	Format     string `yaml:"format"`
	Codec      string `yaml:"codec"`
}
type Features struct {
	TimeWindow int `yaml:"time_window"`
	Overlap    int `yaml:"overlap"`
}
type Root struct {
	Pipeline struct {
		Name    string `yaml:"name"`
		Version string `yaml:"version"`
		LogLvl  string `yaml:"log_level"`
	} `yaml:"pipeline"`
	Audio    Audio    `yaml:"audio"`
	Services Services `yaml:"services"`
	Features Features `yaml:"features"`
	Paths    struct {
		Data    string `yaml:"data"`
		Models  string `yaml:"models"`
		Outputs string `yaml:"outputs"`
	} `yaml:"paths"`
}

func Load() (*Root, error) {
	env := os.Getenv("CONFIG_ENV")
	if env == "" {
		env = "dev"
	}
	var guess []string = []string{
		filepath.Join("config", env, "config.yaml"),
		filepath.Join("src", "shared", "config.yaml"),
	}
	var f *os.File
	var err error
	for _, p := range guess {
		f, err = os.Open(p)
		if err == nil {
			defer f.Close()
			var cfg Root
			if err = yaml.NewDecoder(f).Decode(&cfg); err == nil {
				return &cfg, nil
			}
		}
	}
	return nil, err
}

func DurSeconds(n int) time.Duration { return time.Duration(n) * time.Second }
