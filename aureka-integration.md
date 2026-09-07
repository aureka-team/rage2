## Recommended implementation order

1. Decide how RAGE will authenticate to Redis.
2. Create the rage-api ECR repository and verify the GitHub OIDC role.
3. Add and run the RAGE Docker publication workflow.
4. Add the rage-api manifests and encrypted secret to gitops-lupai.
5. Validate with kustomize build prod/lupai-prod.
6. Push GitOps changes and verify the Flux HelmRelease, pod probes, service, Qdrant connectivity, Redis connectivity, and one real collection/retrieval request.
