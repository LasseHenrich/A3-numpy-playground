install:
	git init
	uv sync

clean:
	rm -rf .venv .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

test_all:
	uv run python -m pytest tests/

submit_all submit:
	uv run python submit.py

submit_a submit_movement submit_a_movement:
	uv run python submit.py a

submit_b submit_transpose submit_b_transpose:
	uv run python submit.py b

submit_c submit_elementwise submit_c_elementwise:
	uv run python submit.py c

submit_d submit_reduce submit_d_reduce:
	uv run python submit.py d

submit_e submit_broadcasting submit_e_broadcasting:
	uv run python submit.py e

submit_f submit_advanced_broadcasting submit_f_advanced_broadcasting:
	uv run python submit.py f

submit_g submit_statistics submit_g_statistics:
	uv run python submit.py g

submit_h submit_geometry submit_h_geometry:
	uv run python submit.py h

submit_i submit_vectorisation submit_i_vectorisation:
	uv run python submit.py i

submit_j submit_challenge submit_j_challenge:
	uv run python submit.py j

test_a test_movement test_a_movement:
	uv run python -m pytest tests/test_a_movement.py

test_b test_transpose test_b_transpose:
	uv run python -m pytest tests/test_b_transpose.py

test_c test_elementwise test_c_elementwise:
	uv run python -m pytest tests/test_c_elementwise.py

test_d test_reduce test_d_reduce:
	uv run python -m pytest tests/test_d_reduce.py

test_e test_broadcasting test_e_broadcasting:
	uv run python -m pytest tests/test_e_broadcasting.py

test_f test_advanced_broadcasting test_f_advanced_broadcasting:
	uv run python -m pytest tests/test_f_advanced_broadcasting.py

test_g test_statistics test_g_statistics:
	uv run python -m pytest tests/test_g_statistics.py

test_h test_geometry test_h_geometry:
	uv run python -m pytest tests/test_h_geometry.py

test_i test_vectorisation test_i_vectorisation:
	uv run python -m pytest tests/test_i_vectorisation.py

test_j test_challenge test_j_challenge:
	uv run python -m pytest tests/test_j_challenge.py

.PHONY: install clean test_all submit_all submit \
        submit_a submit_movement submit_a_movement \
        submit_b submit_transpose submit_b_transpose \
        submit_c submit_elementwise submit_c_elementwise \
        submit_d submit_reduce submit_d_reduce \
        submit_e submit_broadcasting submit_e_broadcasting \
        submit_f submit_advanced_broadcasting submit_f_advanced_broadcasting \
        submit_g submit_statistics submit_g_statistics \
        submit_h submit_geometry submit_h_geometry \
        submit_i submit_vectorisation submit_i_vectorisation \
        submit_j submit_challenge submit_j_challenge \
        test_a test_movement test_a_movement \
        test_b test_transpose test_b_transpose \
        test_c test_elementwise test_c_elementwise \
        test_d test_reduce test_d_reduce \
        test_e test_broadcasting test_e_broadcasting \
        test_f test_advanced_broadcasting test_f_advanced_broadcasting \
        test_g test_statistics test_g_statistics \
        test_h test_geometry test_h_geometry \
        test_i test_vectorisation test_i_vectorisation \
        test_j test_challenge test_j_challenge \
        compile_grader

compile_grader:
	uv run python grader/compile_grader.py
