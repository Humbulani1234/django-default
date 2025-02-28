# Generated by Django 4.2.4 on 2023-09-30 16:20

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Item",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("menu", models.CharField(max_length=20)),
            ],
            bases=(models.Model, object),
        ),
    ]
