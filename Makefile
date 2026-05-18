.PHONY: build test lint format format-check ci clean

SOURCES := Sources Tests Package.swift

build:
	swift build

test:
	swift test

lint:
	swift format lint --strict --recursive $(SOURCES)
	swiftlint lint --strict

format:
	swift format format --in-place --recursive $(SOURCES)

format-check:
	swift format lint --strict --recursive $(SOURCES)

ci: lint test

clean:
	swift package clean
	rm -rf .build
