{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
    nixpkgs-python.inputs = { nixpkgs.follows = "nixpkgs"; };

    devenv-root = {
      url =  "file+file:///dev/null";
      flake = false;
    };
  };  
  outputs = inputs@{ flake-parts, devenv-root, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];

      perSystem = { lib, config, self', inputs', pkgs, system, ... }:
        {
          devenv.shells.default = {
            name = "last_passing_rank";
            devenv.root =
              let
                devenvRootFileContent = builtins.readFile devenv-root.outPath;
              in
              pkgs.lib.mkIf (devenvRootFileContent != "") devenvRootFileContent;

            languages.python = {
              enable = true;
              version = "3.10.1";
              venv = {
                enable = true;
                # requirements = "${./requirements.txt}";
              };
            };
          };
        };
    };
}
