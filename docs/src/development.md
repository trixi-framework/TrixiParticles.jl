# [Development](@id development)


## Preview of the documentation

To generate the Documentation, first instantiate the `docs` environment
by executing the following command from the TrixiParticles.jl root directory:
```bash
julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"
```
This command only has to be run once. After that, maintain the `docs` environment
as described under [Installation](@ref installation-issues).

With an instantiated `docs` environment, generate the docs with the following command (again from the TrixiParticles.jl root directory):
```bash
julia --project=docs --color=yes docs/make.jl
```
You can then open the generated files in `docs/build` with your webbrowser.
Alternatively, run
```bash
python3 -m http.server -d docs/build
```
and open `localhost:8000` in your webbrowser.


## Release management

To create a new release for TrixiParticles.jl, perform the following steps:
1) Make sure that all PRs and changes that you want to go into the release are merged to
   `main` and that the latest commit on `main` has passed all CI tests.
2) Determine the currently released version of TrixiParticles.jl, e.g., on the
   [release page](https://github.com/trixi-framework/TrixiParticles.jl/releases). For this manual,
   we will assume that the latest release was `v0.2.3`.
3) Decide on the next version number. We follow [semantic versioning](https://semver.org/),
   thus each version is of the form `vX.Y.Z` where `X` is the major version, `Y` the minor
   version, and `Z` the patch version. In this manual, we assume that the major version is
   always `0`, thus the decision process on the new version is as follows:
   * If the new release contains *breaking changes* (i.e., user code might not work as
     before without modifications), increase the *minor* version by one and set the
     *patch* version to zero. In our example, the new version should thus be `v0.3.0`.
   * If the new release only contains minor modifications and/or bug fixes, the *minor*
     version is kept as-is and the *patch* version is increased by one. In our example, the
     new version should thus be `v0.2.4`.
4) Edit the `version` string in the
   [`Project.toml`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/Project.toml)
   and set it to the new version. Push/merge this change to `main`.
5) Go to GitHub and add a comment to the commit that you would like to become the new
   release (typically this will be the commit where you just updated the version). You can
   comment on a commit by going to the
   [commit overview](https://github.com/trixi-framework/TrixiParticles.jl/commits/main/) and clicking
   on the title of the commit. The comment should contain the following text:
   ```
   @JuliaRegistrator register
   ```
6) Wait for the magic to happen! Specifically, JuliaRegistrator will create a new PR to the
   Julia registry with the new release information. After a grace period of ~15 minutes,
   this PR will be merged automatically. A short while after,
   [TagBot](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/.github/workflows/TagBot.yml)
   will create a new release of TrixiParticles.jl in our GitHub repository.
7) Once the new release has been created, the new version can be obtained through the Julia
   package manager as usual.
8) To make sure people do not mistake the latest state of `main` as the latest release, we
   set the version in the `Project.toml` to a *development* version. The development version
   should be the latest released version, with the patch version incremented by one, and the
   `-dev` suffix added. For example, if you just released `v0.3.0`, the new development
   version should be `v0.3.1-dev`. If you just released `v0.2.4`, the new development
   version should be `v0.2.5-dev`.

