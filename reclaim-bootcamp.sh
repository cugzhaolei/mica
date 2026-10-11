#!/bin/bash
# Reclaim Bootcamp partition (nvme0n1p4, 206G NTFS) -> ext4 /mnt/models
# Run as: sudo setsid /mnt/windows/Dev/deepseek4-strata/reclaim-bootcamp.sh
set -ex
umount /mnt/bc
umount /dev/nvme0n1p4 || true
parted -s /dev/nvme0n1 rm 4
parted -s /dev/nvme0n1 mkpart models ext4 765047MiB 976149MiB
partprobe /dev/nvme0n1
sleep 3
mkfs.ext4 -q -L models /dev/nvme0n1p4
mkdir -p /mnt/models
mount /dev/nvme0n1p4 /mnt/models
echo "MODELS_MOUNTED: $(df -h /mnt/models | tail -1)"
