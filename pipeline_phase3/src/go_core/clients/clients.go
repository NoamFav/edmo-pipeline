package clients

import (
	"net"
	"net/http"
	"time"
)

type HTTP struct{ c *http.Client }

func NewHTTP() *HTTP {
	tr := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   1 * time.Minute,
			KeepAlive: 3 * time.Minute,
		}).DialContext,
		MaxIdleConns:          128,
		MaxIdleConnsPerHost:   32,
		IdleConnTimeout:       2 * time.Minute,
		TLSHandshakeTimeout:   3 * time.Minute,
		ExpectContinueTimeout: 1 * time.Minute,
		ResponseHeaderTimeout: 5 * time.Minute,
	}
	return &HTTP{
		c: &http.Client{
			Transport: tr,
			Timeout:   30 * time.Minute,
		},
	}
}
