CREATE TABLE specifications(
	"id" serial PRIMARY KEY,
	"name" varchar,
	"description" varchar,
	"dimensions" varchar[],
	"unit" varchar,
	"suggested_range" varchar[],
	"absolute_range" varchar[]
);