//! Lossless OpenAI Chat history replay without promoting unparsed calls to executable calls.
use rig_core::message::{AssistantContent, ToolCall, ToolCallId, ToolFunction};
use rig_core::providers::openai::completion::{Message, assistant_content_to_messages};
use serde_json::{Value, json};

#[test]
fn raw_arguments_replay_exactly_and_stay_unparsed() -> Result<(), Box<dyn std::error::Error>> {
    for raw in [
        "{\"query\":",
        "",
        " ",
        "null",
        "\"a JSON string\"",
        "\n\\not-json\"😺",
    ] {
        let part = AssistantContent::UnparsedToolCall(ToolCall::new(
            ToolCallId::new_or_mint("call-raw"),
            ToolFunction::new("lookup".into(), raw.to_owned()),
        ));
        let saved = serde_json::to_string(&part)?;
        let restored: AssistantContent = serde_json::from_str(&saved)?;
        if restored != part {
            return Err("generic raw history changed".into());
        }
        if matches!(restored, AssistantContent::ToolCall(_)) {
            return Err("raw history became executable".into());
        }
        let wire = serde_json::to_value(assistant_content_to_messages(vec![restored])?)?;
        if wire.pointer("/0/tool_calls/0/id") != Some(&json!("call-raw")) {
            return Err("wire value changed".into());
        }
        if wire.pointer("/0/tool_calls/0/function/name") != Some(&json!("lookup")) {
            return Err("wire value changed".into());
        }
        if wire.pointer("/0/tool_calls/0/function/arguments") != Some(&json!(raw)) {
            return Err("wire value changed".into());
        }
    }
    Ok(())
}

#[test]
fn structured_arguments_keep_json_meaning() -> Result<(), Box<dyn std::error::Error>> {
    for args in [
        json!({"x":1}),
        json!("a JSON string"),
        json!([1, "x"]),
        json!(null),
        json!(false),
    ] {
        let part = AssistantContent::tool_call("call-json", "lookup", args.clone());
        let wire = serde_json::to_value(assistant_content_to_messages(vec![part])?)?;
        let argument_text = wire
            .pointer("/0/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
            .ok_or("argument string absent")?;
        if serde_json::from_str::<Value>(argument_text)? != args {
            return Err("JSON argument meaning changed".into());
        }
    }
    Ok(())
}

#[test]
fn reasoning_only_and_existing_assistant_controls() -> Result<(), Box<dyn std::error::Error>> {
    for (text, reasoning, tool) in [
        (false, true, false),
        (true, false, false),
        (false, false, true),
        (true, true, false),
        (false, true, true),
        (false, false, false),
    ] {
        let mut parts = Vec::new();
        if reasoning {
            parts.push(AssistantContent::reasoning(" exact reasoning\n"));
        }
        if text {
            parts.push(AssistantContent::text("exact text"));
        }
        if tool {
            parts.push(AssistantContent::tool_call("call-1", "lookup", json!({})));
        }
        let messages = assistant_content_to_messages(parts)?;
        if !(text || reasoning || tool) {
            if !messages.is_empty() {
                return Err("empty assistant emitted a message".into());
            }
            continue;
        }
        if messages.len() != 1 {
            return Err("assistant message lost".into());
        }
        let wire = serde_json::to_value(messages)?;
        if reasoning && wire.pointer("/0/reasoning_content") != Some(&json!(" exact reasoning\n")) {
            return Err("wire value changed".into());
        }
        if text && wire.pointer("/0/content/0/text") != Some(&json!("exact text")) {
            return Err("wire value changed".into());
        }
        if tool && wire.pointer("/0/tool_calls/0/id") != Some(&json!("call-1")) {
            return Err("wire value changed".into());
        }
    }
    Ok(())
}

#[test]
fn malformed_provider_arguments_still_fail_strict_decoding() {
    let wire = json!({"role":"assistant","content":[],"tool_calls":[{"type":"function","id":"call-bad","function":{"name":"lookup","arguments":"{\"x\":"}}]});
    assert!(serde_json::from_value::<Message>(wire).is_err());
}
