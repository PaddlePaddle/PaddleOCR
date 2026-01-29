#!/usr/bin/env bash

set -euo pipefail

device_type='nvidia-gpu'
backend='vllm'
build_for_offline='false'
paddleocr_version='3.4.0'
paddlex_version='3.4.0'
platform='linux/amd64'
action='load'
registry='ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle'
builder=''

show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --device-type <type>      Device type (nvidia-gpu|nvidia-gpu-sm120|hygon-dcu|kunlunxin-xpu|metax-gpu|iluvatar-gpu|huawei-npu) [default: ${device_type}]
  --backend <backend>       Backend type (vllm|fastdeploy) [default: ${backend}]
  --offline                 Build offline version
  --ppocr-version <ver>     PaddleOCR version [default: ${paddleocr_version}]
  --pdx-version <ver>       PaddleX version [default: ${paddlex_version}]
  --platform <platform>     Build platform [default: ${platform}]
  --action <action>         Post-build action: load|push|tar|none [default: ${action}]
                            load: Load to local Docker
                            push: Push to image registry
                            tar: Export as tar file
                            none: Build only, no output
  --registry <registry>     Custom image registry [default: ${registry}]
  --builder <name>          Buildx builder name (override default)
  -h, --help               Show this help message

Examples:
  $0 --device-type nvidia-gpu --backend vllm --action push
  $0 --platform linux/amd64,linux/arm64 --action push
  $0 --action tar --platform linux/amd64
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --device-type)
            [ -z "$2" ] && {
                echo "Error: '--device-type' requires a value" >&2
                exit 2
            }
            device_type="$2"
            shift
            shift
            case "${device_type}" in
                nvidia-gpu|nvidia-gpu-sm120|hygon-dcu|kunlunxin-xpu|metax-gpu|iluvatar-gpu|huawei-npu)
                    ;;
                *)
                    echo "Error: Unknown device type: ${device_type}" >&2
                    exit 2
                    ;;
            esac
            ;;
        --backend)
            [ -z "$2" ] && {
                echo "Error: '--backend' requires a value" >&2
                exit 2
            }
            backend="$2"
            shift 2
            case "${backend}" in
                vllm|fastdeploy)
                    ;;
                *)
                    echo "Error: Unknown backend: ${backend}" >&2
                    exit 2
                    ;;
            esac
            ;;
        --offline)
            build_for_offline='true'
            shift
            ;;
        --ppocr-version)
            [ -z "$2" ] && {
                echo "Error: '--ppocr-version' requires a value" >&2
                exit 2
            }
            paddleocr_version="$2"
            shift
            shift
            ;;
        --pdx-version)
            [ -z "$2" ] && {
                echo "Error: '--pdx-version' requires a value" >&2
                exit 2
            }
            paddlex_version="$2"
            shift
            shift
            ;;
        --platform)
            [ -z "$2" ] && {
                echo "Error: '--platform' requires a value" >&2
                exit 2
            }
            platform="$2"
            shift
            shift
            ;;
        --action)
            [ -z "$2" ] && {
                echo "Error: '--action' requires a value" >&2
                exit 2
            }
            action="$2"
            shift
            shift
            case "${action}" in
                load|push|tar|none)
                    ;;
                *)
                    echo "Error: Unknown action: ${action}. Use load|push|tar|none" >&2
                    exit 2
                    ;;
            esac
            ;;
        --registry)
            [ -z "$2" ] && {
                echo "Error: '--registry' requires a value" >&2
                exit 2
            }
            registry="$2"
            shift
            shift
            ;;
        --builder)
            [ -z "${2-}" ] && {
                echo "Error: '--builder' requires a value" >&2
                exit 2
            }
            builder="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "Use $0 --help for usage information" >&2
            exit 2
            ;;
    esac
done

# Validate platform compatibility for load action
if [[ "${action}" == 'load' ]] && [[ "${platform}" == *','* ]]; then
    echo "Error: Cannot use --action load with multiple platforms" >&2
    echo "Platform: ${platform}" >&2
    echo "Please specify a single platform or use other actions" >&2
    exit 2
fi

# Build tag suffix
tag_suffix="latest-${device_type}"

if [ "${build_for_offline}" = 'true' ]; then
    tag_suffix="${tag_suffix}-offline"
fi

dockerfile="accelerators/${device_type}/vlm.Dockerfile"
if [ ! -f "${dockerfile}" ]; then
    echo "Error: Dockerfile not found: ${dockerfile}" >&2
    exit 1
fi

dockerfile_hash="$(sha256sum "${dockerfile}" | cut -c1-12)"
image_version="${paddleocr_version}-${paddlex_version}-${dockerfile_hash}"

# Image name
base_image_name="paddleocr-genai-${backend}-server"

# Main tags
main_tag="${registry}/${base_image_name}:${tag_suffix}"
version_tag="${registry}/${base_image_name}:${tag_suffix/latest/${image_version}}"
paddleocr_version_tag="${registry}/${base_image_name}:${tag_suffix/latest/paddleocr${paddleocr_version%.*}}"

# Build arguments array
build_args=(
    '--platform' "${platform}"
    '-f' "${dockerfile}"
    '-t' "${main_tag}"
    '-t' "${version_tag}"
    '-t' "${paddleocr_version_tag}"
    '--build-arg' "BUILD_FOR_OFFLINE=${build_for_offline}"
    '--build-arg' "PADDLEOCR_VERSION===${paddleocr_version}"
    '--build-arg' "PADDLEX_VERSION===${paddlex_version}"
    '--build-arg' "BACKEND=${backend}"
    '--build-arg' "http_proxy=${http_proxy:-}"
    '--build-arg' "https_proxy=${https_proxy:-}"
    '--build-arg' "no_proxy=${no_proxy:-}"
    '--label' "org.opencontainers.image.version.paddleocr=${paddleocr_version}"
    '--label' "org.opencontainers.image.version.paddlex=${paddlex_version}"
    '--label' "org.opencontainers.image.version.dockerfile.sha=${dockerfile_hash}"
    '.'
)

if [[ -n "${builder}" ]]; then
    build_args=('--builder' "${builder}" "${build_args[@]}")
fi

echo "========================================="
echo "Build Configuration:"
echo "  Device Type:     ${device_type}"
echo "  Backend:         ${backend}"
echo "  Offline Mode:    ${build_for_offline}"
echo "  PaddleOCR:       ${paddleocr_version}"
echo "  PaddleX:         ${paddlex_version}"
echo "  Platform:        ${platform}"
echo "  Action:          ${action}"
echo "  Registry:        ${registry}"
echo "  Builder:         ${builder:-<default>}"
echo "  Dockerfile:      ${dockerfile}"
echo "  Tags:"
echo "    - ${main_tag}"
echo "    - ${version_tag}"
echo "========================================="

# Add action-specific build options
case "${action}" in
    load)
        echo "Building and loading to local Docker..."
        build_args+=('--load')
        ;;
    push)
        echo "Building and pushing to registry..."
        build_args+=('--push')
        if ! docker info 2>/dev/null | grep -q "Username"; then
            echo "Warning: Docker login not detected. Push may fail."
            echo "Tip: Run 'docker login ${registry%%/*}' to login"
        fi
        ;;
    tar)
        echo "Building and exporting as tar file..."
        output_dir=./build-output
        mkdir -p "${output_dir}"
        tar_file="${output_dir}/${base_image_name}-${tag_suffix}-${platform//\//_}.tar"
        build_args+=('--output' "type=docker,dest=${tar_file}")
        echo "  Output file: ${tar_file}"
        ;;
    none)
        echo "Building only (image stays in BuildKit cache)..."
        echo "Note: Image will be lost if BuildKit cache is cleared"
        ;;
esac

# Execute build command
echo "Executing: docker buildx build ${build_args[*]}"
docker buildx build "${build_args[@]}"

# Post-build actions
case "${action}" in
    load)
        echo "✅ Image loaded to local Docker"
        echo "   View: docker images | grep ${base_image_name}"
        ;;
    push)
        echo "✅ Image pushed to registry"
        echo "   Pull: docker pull ${main_tag}"
        echo "   Inspect: docker buildx imagetools inspect ${main_tag}"
        ;;
    tar)
        echo "✅ Image exported as tar file"
        echo "   File size: $(du -h "${tar_file}" | cut -f1)"
        echo "   Load to Docker: docker load -i ${tar_file}"
        ;;
    none)
        echo "ℹ️  Image saved in BuildKit cache only"
        echo "   Use --action load to rebuild and load quickly"
        echo "   Or use --action push to push directly"
        ;;
esac
