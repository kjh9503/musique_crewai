# musique_crewai

This repository provides a CrewAI-based pipeline for running and evaluating experiments on the **Musique dataset**.

## Usage

1. Use the sample data in the `./data` directory (`musique_train_x.json`),  
   or download the original dataset from  
   [Musique Dataset (Google Drive)](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view)  
   and preprocess it using `./data/process.py`.

2. Run `./run_all.sh`.

3. Check the generated result files in the `output` directory  
   (`answer_x.md`, `answer_kg_x.md`, `answer_subquestions_x.md`).

4. Evaluate the results using `./eval_answers.py`.
