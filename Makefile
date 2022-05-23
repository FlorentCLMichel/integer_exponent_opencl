build:
	cargo build --release --offline

test: 
	cargo test --offline

debug: 
	cargo build --offline

clippy:
	cargo clippy --offline

clean: 
	rm -rf target
	rm -f Cargo.lock
