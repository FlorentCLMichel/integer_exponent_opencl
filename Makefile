build:
	cargo build --release --offline

test: 
	cargo test --offline

debug: 
	cargo build --offline

clean: 
	rm -rf target
	rm -f Cargo.lock
