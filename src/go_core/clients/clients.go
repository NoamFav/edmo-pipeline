package clients

import (
	"net/http"
	"time"
)

type HTTP struct{ c *http.Client }

func NewHTTP() *HTTP { return &HTTP{c: &http.Client{Timeout: 60 * time.Second}} }
