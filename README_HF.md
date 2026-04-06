---
title: DPN Classification API
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DPN Classification API

REST API for Diabetic Peripheral Neuropathy (DPN) detection from plantar thermograms.

## Mobile App Endpoint

```
POST /predict/patient/mobile
```

Send a JSON body with base64-encoded thermal images and temperature matrices for both feet.

## Health Check

```
GET /health
```

## API Docs

```
GET /docs
```
