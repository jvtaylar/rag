if ask_button:
    if not st.session_state.openai_api_key:
        st.warning("Set your OpenAI API key in the sidebar first.")
    elif not user_question:
        st.warning("Please type a question.")
    else:
        openai.api_key = st.session_state.openai_api_key

        # Case 1: RAG (documents uploaded)
        if st.session_state.store:
            with st.spinner("Retrieving..."):
                q_emb = get_embeddings(openai, [user_question], model=embedding_model)[0]
                top_chunks = retrieve_top_k(q_emb, st.session_state.store, k=int(top_k))

            context = "\n\n".join(
                [f"Source: {c['metadata']['source']} (chunk {c['metadata']['chunk_index']})\n{c['content']}" for c in top_chunks]
            )

            system_prompt = (
                "You are a helpful assistant. Use the provided CONTEXT to answer the user question. "
                "If the answer is not contained in the context, say you don't know and avoid hallucination. "
                "Cite the source filename and chunk index when referencing the document."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n---\nQUESTION:\n{user_question}"},
            ]

        # Case 2: No documents uploaded → answer normally
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question as best as you can."},
                {"role": "user", "content": user_question},
            ]

        # Generate response
        with st.spinner("Generating answer from the LLM..."):
            try:
                resp = openai.ChatCompletion.create(
                    model=st.session_state.get('chat_model', chat_model),
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                answer = None

        if answer:
            st.markdown("### Answer")
            st.write(answer)

        # Show snippets only if docs were used
        if st.session_state.store:
            st.markdown("---")
            st.markdown("### Retrieved snippets (ranked)")
            for i, c in enumerate(top_chunks, start=1):
                st.write(
                    f"**{i}. Source:** {c['metadata']['source']} "
                    f"(chunk {c['metadata']['chunk_index']}) — score: {c.get('score', 0):.4f}"
                )
                st.write(c['content'][:1000] + ("..." if len(c['content']) > 1000 else ""))
