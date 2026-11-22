/* ds41_chat_test -- DeepSeek-V4.1's chat template with pictures, against the release's encoding/encoding.py.
 *
 * DS41_VIS_REF/chat_cases.json (written by ~/gpu/ds41/vis/scripts/ref_prompts.py from encoding.py, unchanged) holds
 * requests OpenAI-shaped as the server receives them and encode_messages()'s prompt for each. The server turns every
 * picture part into a media_marker part (oai.cpp lower_content_parts); the template renders the marker where
 * encoding.py renders <｜deepseek_image｜>, joined to the other blocks the same way ("\n\n"), so with the marker
 * replaced by the placeholder the two prompts are byte-identical. DS41_VIS_TEMPLATE is the template to test and
 * DS41_VIS_TEMPLATE_OLD the one the published container carries: text-only requests render identically through
 * both. Without the variables every case skips. */
#include "rad_test.h"

#include "text/chat.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using json = nlohmann::json;
using namespace rad;

namespace {
std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
std::string env(const char* k) {
    const char* v = std::getenv(k);
    return v ? v : "";
}
const char* kMarker = "<__media__>";
const char* kImg = "<｜deepseek_image｜>";

/* Picture parts as the server lowers them. */
json lower(json msgs) {
    for (auto& m : msgs) {
        if (!m.contains("content") || !m["content"].is_array()) continue;
        for (auto& p : m["content"])
            if (p.value("type", "") == "image_url" || p.value("type", "") == "image")
                p = json{ { "type", "media_marker" }, { "text", kMarker } };
    }
    return msgs;
}

bool render(const std::string& src, const json& c, json msgs, std::string* out, std::string* err) {
    ChatTemplate t;
    if (t.load(src, "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>") < 0 || !t.ready()) {
        *err = "the template did not load";
        return false;
    }
    ChatOptions opt;
    opt.quiet = true;
    opt.add_generation_prompt = true;
    const json& kw = c["kwargs"];
    opt.enable_thinking = kw.value("thinking_mode", "chat") == "thinking";
    for (const auto& [k, v] : kw.items()) opt.template_kwargs[k] = v.dump();
    ChatPrompt p;
    const json tools = c.contains("tools") && c["tools"].is_array() ? c["tools"] : json();
    if (t.apply(msgs, tools, opt, &p) < 0) {
        *err = p.error;
        return false;
    }
    *out = p.prompt;
    return true;
}
std::string first_difference(const std::string& want, const std::string& got) {
    size_t i = 0;
    while (i < want.size() && i < got.size() && want[i] == got[i]) ++i;
    const size_t from = i > 40 ? i - 40 : 0;
    return "at byte " + std::to_string(i) + "\n      reference: " + json(want.substr(from, 120)).dump() +
           "\n      engine:    " + json(got.substr(from, 120)).dump();
}
}  // namespace

TEST(ds41_template_renders_pictures_as_encoding_py) {
    const std::string ref = env("DS41_VIS_REF"), tpl = env("DS41_VIS_TEMPLATE");
    if (ref.empty() || tpl.empty()) SKIP("DS41_VIS_REF / DS41_VIS_TEMPLATE are not set");
    const json cases = json::parse(slurp(ref + "/chat_cases.json"));
    const std::string src = slurp(tpl);
    int n = 0;
    for (const json& c : cases) {
        const std::string name = c["name"];
        if (name.rfind("text_only_", 0) == 0) continue;
        std::string got, err;
        const bool ok = render(src, c, lower(c["messages"]), &got, &err);
        if (!ok) fprintf(stderr, "    %s: refused: %s\n", name.c_str(), err.c_str());
        CHECK(ok);
        size_t markers = 0;
        for (size_t at; (at = got.find(kMarker)) != std::string::npos; ++markers)
            got.replace(at, std::strlen(kMarker), kImg);
        const std::string want = c["prompt"];
        const bool same = got == want;
        fprintf(stderr, "    %s: %zu picture(s), %zu bytes, %s\n", name.c_str(), markers, got.size(),
                same ? "identical to encoding.py" : ("DIFFERS " + first_difference(want, got)).c_str());
        CHECK(same);
        ++n;
    }
    CHECK(n >= 5);
}

TEST(ds41_template_text_renders_unchanged) {
    const std::string ref = env("DS41_VIS_REF"), tpl = env("DS41_VIS_TEMPLATE"), old = env("DS41_VIS_TEMPLATE_OLD");
    if (ref.empty() || tpl.empty() || old.empty()) SKIP("DS41_VIS_REF / _TEMPLATE / _TEMPLATE_OLD are not set");
    const json cases = json::parse(slurp(ref + "/chat_cases.json"));
    const std::string a = slurp(tpl), b = slurp(old);
    int n = 0;
    for (const json& c : cases) {
        const std::string name = c["name"];
        if (name.rfind("text_only_", 0) != 0) continue;
        for (const char* mode : { "chat", "thinking" }) {
            json cc = c;
            cc["kwargs"]["thinking_mode"] = mode;
            std::string ra, rb, ea, eb;
            const bool oa = render(a, cc, cc["messages"], &ra, &ea), ob = render(b, cc, cc["messages"], &rb, &eb);
            CHECK_EQ(oa, ob);
            const bool same = ra == rb && ea == eb;
            fprintf(stderr, "    %s (%s): %zu bytes, %s\n", name.c_str(), mode, ra.size(),
                    same ? "identical through both templates" : "DIFFERS");
            CHECK(same);
            ++n;
        }
    }
    CHECK(n >= 8);
}

RAD_TEST_MAIN()
