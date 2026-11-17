/* node_rendezvous.h -- the two nodes agree on what they run before they run it (tp8 S4).
 *
 * Twice over the link, before its threads start. Phase 1 (kHello), before any weight is declared
 * or loaded: the binary and the plugins (file hashes), the image id the launcher names, the model
 * (size and a hash of its head), every RADIANCE_* setting, every runtime switch, every option
 * that is not the node's own, the cut, the link, and each side's role. Phase 2 (kHello2), once both
 * have carved their KV pools and laid out the step batch: the KV groups' geometry and token
 * capacity, the step batch's layout, and each crossing's row bytes as the sender and the receiver
 * declared them. Any difference stops both starts with a line naming it; none is papered over,
 * since one byte of difference in a step is a different answer.
 */
#pragma once
#include "net/node_link.h"

#include <string>
#include <utility>
#include <vector>

namespace rad::net {

/* Ordered (key, value) pairs; encoded as `key\tvalue\n` lines. */
struct Facts {
    std::vector<std::pair<std::string, std::string>> kv;
    void add(const std::string& k, const std::string& v) { kv.emplace_back(k, v); }
    const std::string* find(const std::string& k) const;
    std::string encode() const;
    static bool decode(const std::vector<uint8_t>& body, Facts* out);
};

/* Sends `mine` as a `kind` frame and reads the peer's (both sides send first, so neither waits on
 * the other). RAD_E_IO with the reason in `err`. */
int exchange(NodeLink& link, uint16_t kind, const Facts& mine, Facts* theirs, int timeout_ms,
             std::string* err);

/* Every key either side has whose values differ, as one line each ("key: this node 'a', the
 * peer 'b'"), keys in `own` skipped (a node's own settings: role, address, budgets). Empty when
 * they agree. A key only one side has counts as a difference. */
std::vector<std::string> differences(const Facts& mine, const Facts& theirs,
                                     const std::vector<std::string>& own);

/* A 64-bit identity of a file's bytes (CRC-32C of the whole file, and of its size), "" when it
 * cannot be read; `head` > 0 hashes only the first `head` bytes. */
std::string file_identity(const std::string& path, int64_t head = 0);

/* The shared objects loaded from a radiance home (kernels/, architectures/, quantizers/), each
 * "name=identity", sorted by name. */
std::vector<std::string> loaded_plugin_identities();

}  /* namespace rad::net */
