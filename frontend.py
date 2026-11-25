
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CourseMate AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in {
    "documents": [],
    "selected_document": None,
    "chat_history": {},
    "summaries": {},
    "extracted": {},
    "conversation_ids": {},
    "restored_documents": set(),
    "new_chat_documents": set(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def error_detail(response):
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


if not st.session_state.documents:
    try:
        response = requests.get(f"{API_URL}/documents", timeout=30)
        if response.ok:
            st.session_state.documents = [
                item["filename"] for item in response.json().get("documents", [])
            ]
            if st.session_state.documents and st.session_state.selected_document not in st.session_state.documents:
                st.session_state.selected_document = st.session_state.documents[0]
    except requests.RequestException:
        pass


st.title("CourseMate AI")
st.caption("Your personal AI study assistant — questions search across all uploaded study materials.")

with st.sidebar:
    st.header("Study materials")

    uploaded_files = st.file_uploader(
        "Upload study materials",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Upload documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one document.")
        else:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name

                try:
                    with st.spinner(f"Uploading {filename}..."):
                        response = requests.post(
                            f"{API_URL}/upload",
                            files={
                                "file": (
                                    filename,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type
                                    or "application/octet-stream",
                                )
                            },
                            timeout=180,
                        )

                    if response.ok:
                        if filename not in st.session_state.documents:
                            st.session_state.documents.append(filename)

                        st.session_state.selected_document = filename

                        # A replacement document needs a fresh conversation.
                        st.session_state.chat_history.pop(filename, None)
                        st.session_state.summaries.pop(filename, None)
                        st.session_state.extracted.pop(filename, None)
                        st.session_state.conversation_ids.pop(filename, None)
                        st.session_state.restored_documents.discard(filename)
                        st.session_state.new_chat_documents.add(filename)

                        st.success(f"Uploaded: {filename}")
                    else:
                        st.error(
                            f"{filename}: {error_detail(response)}"
                        )

                except requests.RequestException:
                    st.error(
                        "Could not reach the backend. "
                        "Make sure the FastAPI server is running."
                    )

    if st.session_state.documents:
        st.divider()

        documents = st.session_state.documents
        current = st.session_state.selected_document
        selected_index = (
            documents.index(current) if current in documents else 0
        )

        selected = st.selectbox(
            "Choose a document",
            options=documents,
            index=selected_index,
        )

        if selected != st.session_state.selected_document:
            st.session_state.selected_document = selected
            st.rerun()

        if st.button("New chat", use_container_width=True):
            st.session_state.chat_history[selected] = []
            st.session_state.conversation_ids.pop(selected, None)
            st.session_state.new_chat_documents.add(selected)
            st.rerun()

        if st.button("Generate study summary", use_container_width=True):
            try:
                with st.spinner("Preparing your study summary..."):
                    response = requests.post(
                        f"{API_URL}/summarize",
                        json={"filename": selected},
                        timeout=300,
                    )

                if response.ok:
                    st.session_state.summaries[selected] = (
                        response.json().get("summary", "")
                    )
                else:
                    st.error(error_detail(response))

            except requests.RequestException:
                st.error("Could not reach the backend.")

        if st.button("View extracted text", use_container_width=True):
            try:
                with st.spinner("Loading document text..."):
                    response = requests.get(
                        f"{API_URL}/extract/{selected}",
                        timeout=60,
                    )

                if response.ok:
                    st.session_state.extracted[selected] = response.json()
                else:
                    st.error(error_detail(response))

            except requests.RequestException:
                st.error("Could not reach the backend.")


filename = st.session_state.selected_document

if (
    filename
    and filename not in st.session_state.restored_documents
    and filename not in st.session_state.new_chat_documents
):
    try:
        response = requests.get(
            f"{API_URL}/documents/{filename}/conversations",
            timeout=30,
        )
        if response.ok:
            conversations = response.json().get("conversations", [])
            if conversations:
                latest = conversations[0]
                detail = requests.get(
                    f"{API_URL}/conversations/{latest['id']}",
                    timeout=30,
                )
                if detail.ok:
                    st.session_state.conversation_ids[filename] = latest["id"]
                    st.session_state.chat_history[filename] = [
                        {"role": item["role"], "content": item["content"]}
                        for item in detail.json().get("messages", [])
                        if item["role"] in {"user", "assistant"}
                    ]
    except requests.RequestException:
        pass
    st.session_state.restored_documents.add(filename)

if not filename:
    st.subheader("Your study workspace")
    st.info("Upload a PDF, TXT, or DOCX document from the sidebar to start chatting.")
else:
    st.subheader(f"Chatting about: {filename}")

    if filename not in st.session_state.chat_history:
        st.session_state.chat_history[filename] = []

    history = st.session_state.chat_history[filename]

    if not history:
        st.markdown("### Start studying")
        st.write(
            "Ask a question about your document. "
            "You can ask follow-up questions in the same chat."
        )

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(source)

    question = st.chat_input(
        "Ask a question about your uploaded documents..."
    )

    if question:
        history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "filename": filename,
                            "filenames": st.session_state.documents,
                            "question": question,
                            "conversation_id": st.session_state.conversation_ids.get(filename),
                            # The current question was just appended above.
                            # Send only earlier turns as conversational context.
                            "history": [
                                {
                                    "role": message["role"],
                                    "content": message["content"],
                                }
                                for message in history[:-1][-24:]
                                if message.get("role") in {"user", "assistant"}
                            ],
                        },
                        timeout=180,
                    )

                    if response.ok:
                        result = response.json()
                        answer = result.get(
                            "answer",
                            "No answer was returned.",
                        )
                        st.session_state.conversation_ids[filename] = result.get(
                            "conversation_id"
                        )

                        sources = [
                            f"{source.get('filename', filename)} — page/section {source['page']}"
                            for source in result.get(
                                "source_pages_or_sections", []
                            )
                            if source.get("page") is not None
                        ]

                        st.markdown(answer)

                        if sources:
                            with st.expander("Sources"):
                                for source in sources:
                                    st.write(source)

                        history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                    else:
                        error_message = error_detail(response)
                        st.error(error_message)
                        history.append({
                            "role": "assistant",
                            "content": f"I couldn't answer that: {error_message}",
                        })

                except requests.RequestException:
                    error_message = (
                        "I couldn't connect to the backend. "
                        "Please check that the server is running."
                    )
                    st.error(error_message)
                    history.append({
                        "role": "assistant",
                        "content": error_message,
                    })

        st.rerun()

    summary = st.session_state.summaries.get(filename)
    if summary:
        with st.expander("Study summary"):
            st.markdown(summary)

    extracted = st.session_state.extracted.get(filename)
    if extracted:
        with st.expander("Extracted document text"):
            for section in extracted.get("pages", []):
                section_number = section.get("page", 1)
                st.markdown(f"**Section {section_number}**")
                st.text(section.get("text", ""))
