# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability within scVAE-Annotator, please follow these steps:

### 1. Report via GitHub Security Advisory

Go to the [Security tab](https://github.com/or4k2l/scVAE-Annotator/security/advisories) and click "Report a vulnerability".

### 2. Or Email Us

Send an email to: **security@scvae-annotator.dev** (if available)

### 3. Include the Following Information

Please include as much of the following information as possible:

- **Type of vulnerability** (e.g., buffer overflow, SQL injection, cross-site scripting)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it

### 4. What to Expect

- **Initial Response**: Within 48 hours of your report
- **Status Update**: Within 7 days with a detailed response
- **Resolution Timeline**: We aim to release patches for critical vulnerabilities within 30 days

### 5. Disclosure Policy

- We will coordinate with you regarding the disclosure timeline
- We prefer to fully investigate and patch vulnerabilities before public disclosure
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using scVAE-Annotator:

### Data Security

- **Do not include sensitive data** in issue reports or pull requests
- **Sanitize datasets** before sharing for debugging
- **Use secure file handling** when working with patient or proprietary data

### Dependency Security

We regularly update dependencies to patch known vulnerabilities:

```bash
# Check for known vulnerabilities
safety check

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Running in Production

- **Use virtual environments** to isolate dependencies
- **Keep Python and dependencies updated** to the latest stable versions
- **Restrict file system access** when running on shared systems
- **Validate input data** before processing

## Security Features

scVAE-Annotator includes:

- ✅ **Dependency scanning** via Safety in CI/CD
- ✅ **Code security analysis** via Bandit
- ✅ **Automated dependency updates** (planned)
- ✅ **Input validation** for configuration and data files

## Known Issues

Currently, there are no known security issues. We will list any discovered vulnerabilities here after they have been patched.

## Security Updates

Security updates will be:

1. Released as patch versions (e.g., 0.1.1)
2. Documented in [CHANGELOG.md](CHANGELOG.md)
3. Announced via GitHub Security Advisories
4. Tagged with `security` label in releases

## Contact

For security-related questions that are not vulnerabilities, please open a GitHub issue with the `security` label.

## Attribution

We follow the guidelines from:
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [OWASP Security Guidelines](https://owasp.org/)

---

**Thank you for helping keep scVAE-Annotator and our users safe!**
