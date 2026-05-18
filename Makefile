.PHONY: build dev-build metallib verify-metallib test lint format format-check ci clean

SOURCES := Sources Tests Package.swift
CONFIG ?= release

build:
	swift build -c $(CONFIG)
	$(MAKE) metallib CONFIG=$(CONFIG)

dev-build:
	$(MAKE) build CONFIG=debug

metallib:
	scripts/build-metallib.sh $(CONFIG)

verify-metallib:
	@test -s .build/$(CONFIG)/mlx.metallib \
		|| (echo "missing or empty .build/$(CONFIG)/mlx.metallib" && exit 1)
	@echo "verify-metallib: .build/$(CONFIG)/mlx.metallib present"

test:
	swift test

lint:
	swift format lint --strict --recursive $(SOURCES)
	swiftlint lint --strict

format:
	swift format format --in-place --recursive $(SOURCES)

format-check:
	swift format lint --strict --recursive $(SOURCES)

ci: lint test build verify-metallib

clean:
	swift package clean
	rm -rf .build
