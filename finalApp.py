def AVM(q,k,o):
    openai.api_key = k
    pinecone_key = o
    query =  q
    def getData(path):
        def extract_text_from_pdf(pdf_path):
            text = ""
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            return text

        # Replace "your_pdf_file.pdf" with the actual file name you uploaded
        pdf_path = path
        text = extract_text_from_pdf(pdf_path)
        #print(text)

        def split_text_into_parts(text, num_parts=10):
            # Calculate the length of each part
            part_length = len(text) // num_parts
            # Split the text into parts
            parts = [text[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
            return parts

        # Example usage:
        # Assuming you have already extracted the text and stored it in the 'text' variable
        text_parts = split_text_into_parts(text, num_parts=20)
        return text_parts


    my_list = getData("book-no-1.pdf")

    def getIndex():
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("quickstart")
        return index

    index = getIndex()

    def upserts(query, my_list, index):
        index = index
        my_list = my_list
        query = query
        MODEL = "text-embedding-3-small"

        res = openai.Embedding.create(
            input=[query], engine=MODEL
        )

        embeds = [record['embedding'] for record in res['data']]

        # load the first 1K rows of the TREC dataset
        #trec = load_dataset('trec', split='train[:1000]')

        batch_size = 32  # process everything in batches of 32
        for i in tqdm(range(0, len(my_list), batch_size)):
            # set end position of batch
            i_end = min(i+batch_size, len(my_list))
            # get batch of lines and IDs
            lines_batch = my_list[i: i+batch_size]
            ids_batch = [str(n) for n in range(i, i_end)]
            # create embeddings
            res = openai.Embedding.create(input=lines_batch, engine=MODEL)
            embeds = [record['embedding'] for record in res['data']]
            # prep metadata and upsert batch
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert))

    upserts(query, my_list, index)

    def getRes(query):
        query = query
        MODEL = "text-embedding-3-small"

        xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

        res = index.query(vector = [xq], top_k=5, include_metadata=True)

        return res
    
    def vectorQuotes(query):
        similarity = getRes(query)
        justQuotes = []
        for i in range(len(similarity['matches'])):
            justQuotes.append(similarity['matches'][i]['metadata']['text'])
        return justQuotes
    
    def getFinalSummaryGPT4(my_list, queryContext):
        my_list = my_list
        queryContext = queryContext

        # Function to split a list into equal sublists
        def split_list(lst, num_sublists):
            avg = len(lst) // num_sublists
            remainder = len(lst) % num_sublists
            return [lst[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)] for i in range(num_sublists)]

        # Split 'my_list' into n equal sublists
        n = 5
        sublists = split_list(my_list, n)

        # Generate summaries for each sublist using the OpenAI API
        sublist_summaries = []

        for i, sublist in enumerate(sublists):
            sublist_text = ' '.join(sublist)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.9,
                top_p=0.9,
                messages= [{ "role": "user", "content": queryContext+sublist_text }] )

            # Extract the summary from the API response
            summary = response.choices[0].message.content
            sublist_summaries.append(summary)

        # Combine the 10 summaries into one variable
        combined_summary = ' '.join(sublist_summaries)

        # Add a specific prompt tailored to your data
        specific_prompt = f"Given the following summaries:\n{combined_summary}\n\nGenerate a coherent final summary that captures the essence of the provided information."

        specific_prompt = queryContext + specific_prompt
        # Use OpenAI API to generate the final coherent summary

        response_combined = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.9,
            top_p=0.9,
            messages= [{ "role": "user", "content": specific_prompt}] )

        # Extract the final summary from the API response
        final_summary = response_combined.choices[0].message.content.strip()

        return final_summary
    
    contexts = "Based solely on the following information create a coherent answer to the question"
    justQuotes = vectorQuotes(query)
    queryContext = query + ". " + contexts
    responseQuotes = getFinalSummaryGPT4(justQuotes, queryContext)
    return responseQuotes


output = AVM()



