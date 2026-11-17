/* node_rendezvous.cpp -- see node_rendezvous.h. */
#include "net/node_rendezvous.h"

#include <link.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>

namespace rad::net {

const std::string* Facts::find(const std::string& k) const {
    for (const auto& [a, b] : kv)
        if (a == k) return &b;
    return nullptr;
}

/* Tabs and newlines in a value would break the line form; they become spaces (no fact needs
 * them). */
static std::string clean(const std::string& s) {
    std::string o = s;
    for (char& c : o)
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return o;
}

std::string Facts::encode() const {
    std::string s;
    for (const auto& [k, v] : kv) s += clean(k) + "\t" + clean(v) + "\n";
    return s;
}

bool Facts::decode(const std::vector<uint8_t>& body, Facts* out) {
    out->kv.clear();
    const std::string s(body.begin(), body.end());
    size_t at = 0;
    while (at < s.size()) {
        size_t nl = s.find('\n', at);
        if (nl == std::string::npos) return false;
        const std::string line = s.substr(at, nl - at);
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) return false;
        out->kv.emplace_back(line.substr(0, tab), line.substr(tab + 1));
        at = nl + 1;
    }
    return true;
}

int exchange(NodeLink& link, uint16_t kind, const Facts& mine, Facts* theirs, int timeout_ms,
             std::string* err) {
    const std::string body = mine.encode();
    if (link.send(kind, 0, 0, body.data(), (int64_t)body.size()) < 0) {
        *err = "sending this node's facts: " + link.why();
        return RAD_E_IO;
    }
    Msg m;
    const int st = link.recv(&m, timeout_ms);
    if (st == RAD_E_NOTFOUND) {
        *err = fmt("the peer sent no %s within %d s", frame_name(kind), timeout_ms / 1000);
        return RAD_E_IO;
    }
    if (st < 0) {
        *err = "the peer's facts did not arrive: " + link.why();
        return RAD_E_IO;
    }
    if (m.h.kind != kind) {
        *err = fmt("expected the peer's %s, got a %s frame: the two nodes are at different points of "
                   "their start", frame_name(kind), frame_name(m.h.kind));
        return RAD_E_IO;
    }
    if (!Facts::decode(m.body, theirs)) {
        *err = fmt("the peer's %s is not a list of facts", frame_name(kind));
        return RAD_E_IO;
    }
    return RAD_OK;
}

std::vector<std::string> differences(const Facts& mine, const Facts& theirs,
                                     const std::vector<std::string>& own) {
    auto is_own = [&](const std::string& k) {
        return std::find(own.begin(), own.end(), k) != own.end();
    };
    std::vector<std::string> out;
    for (const auto& [k, v] : mine.kv) {
        if (is_own(k)) continue;
        const std::string* t = theirs.find(k);
        if (!t) out.push_back(fmt("%s: this node '%s', the peer has none", k.c_str(), v.c_str()));
        else if (*t != v)
            out.push_back(fmt("%s: this node '%s', the peer '%s'", k.c_str(), v.c_str(), t->c_str()));
    }
    for (const auto& [k, v] : theirs.kv) {
        if (is_own(k) || mine.find(k)) continue;
        out.push_back(fmt("%s: the peer '%s', this node has none", k.c_str(), v.c_str()));
    }
    return out;
}

std::string file_identity(const std::string& path, int64_t head) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    std::vector<char> buf(1 << 20);
    uint32_t c = 0;
    int64_t total = 0;
    for (;;) {
        size_t want = buf.size();
        if (head > 0) {
            if (total >= head) break;
            want = (size_t)std::min<int64_t>((int64_t)want, head - total);
        }
        const size_t n = std::fread(buf.data(), 1, want, f);
        if (n == 0) break;
        c = crc32c(buf.data(), n, c);
        total += (int64_t)n;
    }
    int64_t size = total;
    if (head > 0 && std::fseek(f, 0, SEEK_END) == 0) size = (int64_t)std::ftell(f);
    std::fclose(f);
    return fmt("%08x-%llu", c, (unsigned long long)size);
}

std::vector<std::string> loaded_plugin_identities() {
    std::map<std::string, std::string> found;
    dl_iterate_phdr([](dl_phdr_info* info, size_t, void* ud) -> int {
        auto* m = static_cast<std::map<std::string, std::string>*>(ud);
        const char* p = info->dlpi_name;
        if (!p || !*p) return 0;
        const std::string s = p;
        for (const char* dir : { "/kernels/", "/architectures/", "/quantizers/" }) {
            const size_t at = s.rfind(dir);
            if (at == std::string::npos) continue;
            (*m)[s.substr(at + 1)] = s;
            break;
        }
        return 0;
    }, &found);
    std::vector<std::string> out;
    for (const auto& [name, path] : found) out.push_back(name + "=" + file_identity(path));
    return out;
}

}  /* namespace rad::net */
