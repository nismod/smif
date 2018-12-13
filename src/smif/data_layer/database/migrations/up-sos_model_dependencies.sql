CREATE TABLE sos_model_dependencies(
	"id" serial PRIMARY KEY,
	"sos_model_name" varchar,
	"sos_model_id" integer,
	"source_model" varchar,
	"source_output" varchar ,
	"sink_model" varchar,
	"sink_input" varchar,
	"lag" varchar
);