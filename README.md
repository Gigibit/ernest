# ernest

## Hugging Face authentication

Place your Hugging Face access token in a local `.env` file to avoid running
`huggingface-cli login`. Supported keys include `HF_TOKEN`,
`HUGGINGFACEHUB_API_TOKEN`, or `HUGGINGFACE_HUB_TOKEN`. Example:

```
HF_TOKEN=hf_xxx
```

The application automatically loads the `.env` file on startup so the token is
available when downloading models.
