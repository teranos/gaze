{
  description = "QNTX gaze plugin — production LLM inference via llama.cpp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # The protocol gaze speaks is QNTX's: plugin/grpc/protocol/*.proto and
    # proto.cmake there. QNTX is taken as source, not as a flake, so none of
    # its inputs are fetched; flake.lock pins the rev.
    qntx = {
      url = "github:teranos/QNTX";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, qntx }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # llama.cpp pinned to the commit CMakeLists.txt fetches when it is not vendored
        llama-cpp-src = pkgs.fetchFromGitHub {
          owner = "ggml-org";
          repo = "llama.cpp";
          rev = "db9d8aa428012cc5593e18635d4c3c54095f5138";
          hash = "sha256-ZX5eaeNZYZIzJyEV3k0Dpcr6L84ccm4YRI++pY9hlJU=";
        };

        gaze-plugin = pkgs.stdenv.mkDerivation {
          pname = "qntx-gaze-plugin";
          version = self.rev or "dev";
          src = ./.;

          postUnpack = ''
            mkdir -p $sourceRoot/vendor
            cp -r ${llama-cpp-src} $sourceRoot/vendor/llama.cpp
          '';

          nativeBuildInputs = with pkgs; [
            cmake
            pkg-config
            grpc
            protobuf
          ];

          buildInputs = with pkgs; [
            grpc
            protobuf
            abseil-cpp
            openssl
            mupdf
          ];

          # In the sandbox llama.cpp does not tune to the machine (GGML_NATIVE is
          # off), and x86 fell to the bare baseline: no SIMD, about 2 tokens a
          # second of prompt. The features named here need Haswell or later.
          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DQNTX_SRC=${qntx}"
          ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isx86_64 [
            "-DGGML_NATIVE=OFF"
            "-DGGML_AVX=ON"
            "-DGGML_AVX2=ON"
            "-DGGML_FMA=ON"
            "-DGGML_F16C=ON"
            "-DGGML_BMI2=ON"
          ];

          # ctest: a build whose tests fail is no build.
          doCheck = true;
        };
      in
      {
        packages = {
          default = gaze-plugin;
          gaze-plugin = gaze-plugin;
        };

        apps.default = {
          type = "app";
          program = "${gaze-plugin}/bin/qntx-gaze-plugin";
        };
      });
}
