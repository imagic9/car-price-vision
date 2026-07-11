# Deployment runbook — car-price-vision demo

Public demo target: **https://cars.pricevision.app** (primary since
2026-07-11; the original **https://aetherkin.space** apex remains a live
alias), served via a **Cloudflare Tunnel** to a FastAPI container on the
deployment VPS. No reverse-proxy involvement, no open inbound ports on the
VPS for this stack. Both hostnames route through the same tunnel: each is an
ingress rule in the tunnel's remote (dashboard/API-managed) config plus a
proxied CNAME to `<tunnel-id>.cfargotunnel.com` in its zone — adding another
domain later needs exactly those two steps and nothing on the VPS.

The rest of this runbook documents the original aetherkin.space bring-up.

The model does not exist yet. The site goes live now as a "coming soon"
placeholder (see `../static/index.html`); the FastAPI app runs and reports
healthy with **no model present**. The trained model is dropped in later
(Phase 5) with zero DNS/tunnel changes -- see step 5.

---

## 1. One-time Cloudflare zone setup (domain owner)

This is a one-time step done by whoever controls the `aetherkin.space`
registration (Namecheap) and email (forwarding via `registrar-servers.com`).

1. In a Cloudflare account (Free plan is enough), choose **Add a site** and
   enter `aetherkin.space`. Cloudflare will auto-scan the existing DNS
   records and stage them for import.
2. **CRITICAL — verify these existing records before continuing, or email
   breaks.** Cloudflare's auto-scan usually finds these but it is not
   guaranteed; check the staged record list and add/fix anything missing:
   - `MX` records at the apex:
     | Priority | Host |
     |---|---|
     | 10 | `eforward1.registrar-servers.com` |
     | 10 | `eforward2.registrar-servers.com` |
     | 10 | `eforward3.registrar-servers.com` |
     | 15 | `eforward4.registrar-servers.com` |
     | 20 | `eforward5.registrar-servers.com` |
   - `TXT` record (SPF) at the apex:
     ```
     v=spf1 include:spf.efwd.registrar-servers.com ~all
     ```
   If any of these are missing from the imported zone, add them manually in
   the Cloudflare DNS dashboard before switching nameservers -- otherwise
   inbound mail forwarding and SPF validation for `aetherkin.space` will
   silently stop working.
3. The current apex **A record** (the existing parked/placeholder site)
   will be **replaced by the tunnel** in step 2, so
   it is safe to delete it (or leave it and let the tunnel's CNAME/A record
   from step 2 override it -- either way, don't leave both pointing at
   conflicting targets).
4. At Namecheap, change the domain's nameservers to the two
   Cloudflare-assigned nameservers shown on the "Add a site" flow (something
   like `xxx.ns.cloudflare.com` / `yyy.ns.cloudflare.com`).
5. Wait for the Cloudflare zone status to flip from "Pending" to "Active"
   (can take a few minutes up to ~24h for nameserver propagation). Do not
   proceed to step 2 until it's Active.

## 2. Create the tunnel (Cloudflare dashboard)

1. Go to **Zero Trust -> Networks -> Tunnels -> Create a tunnel**.
2. Choose the **Cloudflared** connector type.
3. Name it e.g. `car-price-vision`. Create it.
4. On the "Install and run a connector" screen, copy the **tunnel token**
   (the long string after `--token` in the sample command). This is the
   value that goes into `TUNNEL_TOKEN` in `.env` in step 3 -- you do **not**
   need to run the install command shown there; `docker compose` handles
   running `cloudflared` in the container.
5. Continue to **Public Hostname** and add one:
   - **Subdomain**: leave blank (this publishes the apex).
   - **Domain**: `aetherkin.space`
   - **Path**: leave blank.
   - **Service**: Type `HTTP`, URL `car-price-api:8000`
     (this is the Docker service name + port from `docker-compose.yml` in
     this directory -- resolvable because both containers share the
     `car-net` network).
6. Save. Cloudflare creates the corresponding DNS record in the zone from
   step 1 automatically.

## 3. Deploy on the VPS

```bash
# Clone (first time) or pull (subsequent deploys)
git clone <repo-url> car-price-vision   # or: git -C car-price-vision pull
cd car-price-vision/serving/deploy

cp .env.example .env
# edit .env and paste the tunnel token from step 2.4 as TUNNEL_TOKEN=...

docker compose up -d --build
```

Requires the deploying user to have Docker permissions on the VPS (either
`sudo docker ...` or membership in the `docker` group). This stack does not
need root beyond whatever Docker itself requires, and does not modify any
system-level networking (no host ports are published).

## 4. Verify

```bash
docker compose ps
# both car-price-api and car-price-cloudflared should show "Up"/"running"

docker compose logs -f cloudflared
# look for the tunnel registering 4 connections, e.g. lines like:
#   "Registered tunnel connection" connIndex=0
#   ... connIndex=1 / 2 / 3
# (Ctrl+C to stop following)

curl -I https://aetherkin.space/health
# expect: HTTP/2 200
```

Also worth a quick sanity check that the site is really the "coming soon"
placeholder and not erroring:

```bash
curl -s https://aetherkin.space/health
# expect: {"status":"ok","model_loaded":false}
```

## 5. Later (Phase 5): swap in the trained model

No DNS, tunnel, or compose changes needed.

```bash
# on the VPS, from the repo root
cp /path/to/exported/model.onnx models/model.onnx
# optionally also models/model.pt for Grad-CAM support -- see serving/app.py

cd serving/deploy
docker compose restart car-price-api
```

`serving/app.py` picks up the mounted files on startup (`/models` is
read-only inside the container, mapped from `../../models` on the host via
the volume mount in `docker-compose.yml`). Confirm with
`curl -s https://aetherkin.space/health` -> `"model_loaded": true`, and the
landing page will automatically switch from "coming soon" to the live
upload UI on next page load (it checks `/health` on load).

## 6. Operational safety

This stack is fully isolated on its own Docker network (`car-net`) and does
not join, depend on, reference, or otherwise interact with any other
containers already running on the host. `car-price-api`
does not publish any host port (`expose`, not `ports`), so the only way in
is through the Cloudflare Tunnel. Deploying, restarting, or tearing down
this stack (`docker compose up/down/restart` in this directory) must never
stop, restart, or reconfigure any container outside of `car-net`. If a
command you're about to run would touch a container name you don't
recognize from this compose file, stop and double-check before running it.
