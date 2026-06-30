# Quasar - Qwen3-30B-A3B inference engine.
#
# Builds ./quasar: the GGUF loader/inspector, tokenizer, CPU reference forward,
# and the Metal backend. On macOS this links -framework Foundation -framework
# Metal; the binary chooses the CPU or GPU path per run (there is no separate
# CPU-only target).

CC      ?= clang
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?=

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  FRAMEWORKS = -framework Foundation -framework Metal
endif

OBJ = quasar.o gguf.o tokenizer.o chat.o quant.o forward_cpu.o metal.o requant.o \
      json.o sample.o server.o

all: quasar

quasar: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS) $(FRAMEWORKS)

quasar.o: quasar.c quasar.h gguf.h tokenizer.h chat.h forward_cpu.h server.h
	$(CC) $(CFLAGS) -c $< -o $@

json.o: json.c json.h
	$(CC) $(CFLAGS) -c $< -o $@

sample.o: sample.c sample.h
	$(CC) $(CFLAGS) -c $< -o $@

server.o: server.c server.h quasar.h tokenizer.h chat.h forward_cpu.h metal.h sample.h json.h
	$(CC) $(CFLAGS) -c $< -o $@

chat.o: chat.c chat.h
	$(CC) $(CFLAGS) -c $< -o $@

quant.o: quant.c quant.h gguf.h
	$(CC) $(CFLAGS) -c $< -o $@

forward_cpu.o: forward_cpu.c forward_cpu.h quasar.h quant.h gguf.h
	$(CC) $(CFLAGS) -c $< -o $@

requant.o: requant.c requant.h gguf.h quant.h
	$(CC) $(CFLAGS) -c $< -o $@

# Objective-C (Metal backend); ARC manages the Metal objects.
metal.o: metal.m metal.h gguf.h
	$(CC) $(CFLAGS) -fobjc-arc -c metal.m -o metal.o

gguf.o: gguf.c gguf.h
	$(CC) $(CFLAGS) -c $< -o $@

tokenizer.o: tokenizer.c tokenizer.h gguf.h
	$(CC) $(CFLAGS) -c $< -o $@

# Generate a tiny synthetic qwen3moe GGUF and inspect it (no model download).
test-gguf: quasar
	python3 tools/make_test_gguf.py /tmp/quasar_tiny.gguf
	./quasar inspect /tmp/quasar_tiny.gguf

clean:
	rm -f $(OBJ) quasar
	rm -rf *.dSYM

.PHONY: all clean test-gguf
