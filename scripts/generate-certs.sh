#!/usr/bin/env bash
# Generate a self-signed TLS certificate for local HTTPS development.
# On macOS, adds the cert to the system Keychain so Chrome trusts it.
#
# Usage: bash scripts/generate-certs.sh [output_dir]

set -euo pipefail

CERT_DIR="${1:-./certs}"

mkdir -p "$CERT_DIR"

if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
    echo "Certificates already exist in $CERT_DIR — skipping generation."
    echo "  Delete them and re-run this script to regenerate."
    exit 0
fi

echo "Generating self-signed certificate in $CERT_DIR ..."

# Create a proper CA-signed localhost certificate that Chrome will accept.
# Step 1: Generate a local CA key and certificate.
openssl genrsa -out "$CERT_DIR/ca-key.pem" 2048 2>/dev/null

openssl req -new -x509 \
    -key "$CERT_DIR/ca-key.pem" \
    -out "$CERT_DIR/ca-cert.pem" \
    -days 825 \
    -subj "/CN=RAG System Local CA"

# Step 2: Generate the server key and CSR.
openssl genrsa -out "$CERT_DIR/key.pem" 2048 2>/dev/null

openssl req -new \
    -key "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/server.csr" \
    -subj "/CN=localhost"

# Step 3: Sign the server cert with our CA (Chrome requires SAN + max 825 days).
cat > "$CERT_DIR/ext.cnf" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage=digitalSignature,nonRepudiation,keyEncipherment,dataEncipherment
subjectAltName=DNS:localhost,IP:127.0.0.1
EOF

openssl x509 -req \
    -in "$CERT_DIR/server.csr" \
    -CA "$CERT_DIR/ca-cert.pem" \
    -CAkey "$CERT_DIR/ca-key.pem" \
    -CAcreateserial \
    -out "$CERT_DIR/cert.pem" \
    -days 825 \
    -extfile "$CERT_DIR/ext.cnf" \
    2>/dev/null

# Clean up intermediate files (including CA private key — no longer needed).
rm -f "$CERT_DIR/server.csr" "$CERT_DIR/ext.cnf" "$CERT_DIR/ca-cert.srl" "$CERT_DIR/ca-key.pem"

echo "Done. Files created:"
echo "  $CERT_DIR/cert.pem   (server certificate)"
echo "  $CERT_DIR/key.pem    (server private key)"
echo "  $CERT_DIR/ca-cert.pem (local CA — trust this in your OS)"

# On macOS, add the CA cert to the system Keychain so Chrome trusts it.
if [ "$(uname)" = "Darwin" ]; then
    echo ""
    echo "Adding local CA to macOS Keychain (you may be prompted for your password)..."
    sudo security add-trusted-cert -d -r trustRoot \
        -k /Library/Keychains/System.keychain \
        "$CERT_DIR/ca-cert.pem" && \
        echo "CA certificate trusted. Chrome will accept https://localhost:8443" || \
        echo "Failed to add to Keychain. You can manually trust $CERT_DIR/ca-cert.pem via Keychain Access."
fi
