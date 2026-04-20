#!/usr/bin/env bash

pip_build_arg_suffix() {
    local v="$1"
    [[ -z "$v" ]] && {
        echo ""
        return
    }
    if [[ "$v" == '=='* ]]; then
        echo "$v"
        return
    fi
    if [[ "$v" == '>='* ]] || [[ "$v" == '<='* ]] || [[ "$v" == '~='* ]] || [[ "$v" == '!='* ]]; then
        echo "$v"
        return
    fi
    if [[ "$v" == '<'* ]] || [[ "$v" == '>'* ]]; then
        echo "$v"
        return
    fi
    if [[ "$v" =~ ^(https?://|git\+|file://) ]]; then
        echo " @${v}"
        return
    fi
    if [[ "$v" == @* ]]; then
        echo " @${v#@}"
        return
    fi
    echo "==${v}"
}

version_tag_label() {
    local v="$1"
    if [[ "$v" =~ ^(https?://|git\+|file://) ]] || [[ "$v" == @* ]]; then
        if command -v shasum >/dev/null 2>&1; then
            printf 'u%s' "$(printf '%s' "$v" | shasum -a 256 | cut -c1-8)"
        elif command -v sha256sum >/dev/null 2>&1; then
            printf 'u%s' "$(printf '%s' "$v" | sha256sum | cut -c1-8)"
        else
            printf 'u%s' "$(printf '%s' "$v" | cksum | cut -c1-8)"
        fi
    else
        local s="$v"
        [[ "$s" == '=='* ]] && s="${s#==}"
        echo "${s%.*}"
    fi
}
