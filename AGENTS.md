# llama.exp

This is an experimental fork of llama.cpp. Every change must be rebase friendly. Means in standard llama code we implement usually a hook
and the implementation itself comes in one of our classes in the src/moe-hot-cache folder. Follow SoC rules and write tests and comment each method. After that update the html guides a list of links you can find in README.md

### Useful Resources

To conserve context space, load these resources as needed:

- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)
