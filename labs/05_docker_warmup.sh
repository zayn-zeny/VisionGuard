#!/usr/bin/env bash
# labs/05_docker_warmup.sh — Task 6.1 Docker warm-up (before touching the project)
# =================================================================================
# Run a container, map a port, mount a volume, read logs, open a shell inside,
# then list and remove containers and images. Run from the project root:
#   bash labs/05_docker_warmup.sh
#
# Notes for Git Bash on Windows:
#   * MSYS_NO_PATHCONV=1 stops Git Bash from rewriting container paths such as
#     /data into a Windows path before Docker sees them.
#   * `docker exec` is used without -it because the script has no TTY; -t would
#     fail with "cannot attach stdin to a TTY-enabled container".
#   * Host port 8090 is used instead of 8080 so the lab does not collide with
#     whatever already listens on 8080 on this machine.
set -x
export MSYS_NO_PATHCONV=1

HOST_PORT=8090

# 0. Clean up anything left by an earlier interrupted run
docker rm -f vg_warmup_nginx >/dev/null 2>&1 || true
docker volume rm vg_warmup_vol >/dev/null 2>&1 || true

# 1. Run a ready-made image (pulls it, creates + starts a container)
docker run --rm hello-world

# 2. Port mapping + detached container
docker run -d --name vg_warmup_nginx -p ${HOST_PORT}:80 nginx:alpine
docker ps

# 3. Read its logs
docker logs vg_warmup_nginx | tail -5

# 4. Open a shell inside the running container
docker exec vg_warmup_nginx /bin/sh -c "hostname; whoami; ls /usr/share/nginx/html"

# 5. Mount a named volume and write to it from inside the container
docker volume create vg_warmup_vol
docker run --rm -v vg_warmup_vol:/data alpine:3.20 \
    sh -c "echo 'written from inside the container' > /data/proof.txt && cat /data/proof.txt"

# 6. Prove the data persisted in the volume beyond the container's lifetime
docker run --rm -v vg_warmup_vol:/data alpine:3.20 cat /data/proof.txt

# 7. Inspect the published port (host 8090 -> container 80)
docker port vg_warmup_nginx
curl -s -o /dev/null -w "nginx on host port ${HOST_PORT} -> HTTP %{http_code}\n" "http://localhost:${HOST_PORT}"

# 8. Remove the container, volume and images
docker rm -f vg_warmup_nginx
docker volume rm vg_warmup_vol
docker rmi nginx:alpine hello-world alpine:3.20
docker ps -a
docker images

echo "=== DOCKER WARM-UP COMPLETE ==="
