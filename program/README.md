# Shared programs

Programs compose model-independent execution contracts: host service policy,
command schedules, and the adapters that connect them to kernel entry points.
They use `loom_library` providers and can depend on motifs, kernels, and other
programs. A model supplies its identity, parameter layout, preparation, and
selected stages; shared programs cannot depend back on a model package.

[`text/`](text/) contains text-service programs. These are executable service
adapters with explicit host imports and ownership contracts. Their source and
bytecode providers participate in the same model-card link index as numerical
providers, so installed tools can use them without a repository build.
