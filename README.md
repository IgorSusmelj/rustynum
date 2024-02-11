## Usage

### Python bindings

Don't use maturin. But only setup.py

```
cd bindings/python/ &&  python setup.py install
```

or

```
cd bindings/python/ &&  python setup.py bdist_wheel
```

Then run tests using

```
pytest tests
```

or benchmarks using

```
pytest benchmarks
```
