echo "========================================"
echo "Installing TPU-enabled JAX and jaxlib..."
echo "========================================"
# TPU에서 whisper-jax를 실행하기 위해 TPU 지원 jax 버전을 설치합니다.
# (현재 최소 버전은 0.3.20 이상 권장)
pip install --upgrade "jax[tpu]>=0.3.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "========================================"
echo "TPU-enabled JAX installation complete."
echo "========================================"

echo "========================================"
echo "Testing TPU availability in JAX..."
echo "========================================"
python -c "import jax; print('JAX devices:', jax.devices())"